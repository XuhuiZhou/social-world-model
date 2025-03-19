from collections import Counter
import pandas as pd
import json
import os
from tqdm import tqdm
import random
from sotopia.generation_utils import StrOutputParser, agenerate
from typing import Any, Optional
from social_world_model.social_world_model import SocialWorldModel
import asyncio

def str_to_list(s: str) -> list[str]:
    l = s.split(",")
    return [c.strip(" []'") for c in l]

def flatten_fantom_data(entry: dict[str, Any]) -> list[dict[str, Any]]:
    data_list: list[dict[str, Any]] = []
    fact_qa_question = entry['factQA']['question']
    fact_qa_answer = entry['factQA']['correct_answer']
    fact_qa_wrong_answer = entry['factQA']['wrong_answer']
    for key in entry.keys():
        if "QAs" in key:
            for question in entry[key]:
                row = {
                    'question': question['question'],
                    'question_type': question['question_type'],
                    'tom_type': question.get('tom_type', ''),
                    'correct_answer': question['correct_answer'],
                    'wrong_answer': question.get('wrong_answer', ""),
                    'missed_info_accessibility': question['missed_info_accessibility'],
                    'context': entry['short_context'],
                    'full_context': entry['full_context'],
                    'set_id': entry['set_id'],
                    'part_id': entry['part_id'],
                    'complete_question': question['complete_question'],
                    'fact_question': fact_qa_question,
                    'fact_answer': fact_qa_answer,
                }
                data_list.append(row)
        elif "QA" in key and "fact" not in key:
            question = entry[key]
            row = {
                'question': question['question'],
                'question_type': question['question_type'],
                'tom_type': question.get('tom_type', ''),
                'correct_answer': question['correct_answer'],
                'wrong_answer': question.get('wrong_answer', ""),   
                'missed_info_accessibility': question['missed_info_accessibility'],
                'context': entry['short_context'],
                'full_context': entry['full_context'],
                'set_id': entry['set_id'],
                'part_id': entry['part_id'],
                'complete_question': question['complete_question'],
                'fact_question': fact_qa_question,
                'fact_answer': fact_qa_answer,
            }
            data_list.append(row)
        else:            
            continue
    return data_list

class FantomEvalAgent():
    def __init__(self, model_name: str):
        self.model_name = model_name

    def compute_f1(self, ground_truth: str, model_response: str) -> float:
        """
        Compute the F1 score between the ground truth and model response.
        Args:
            ground_truth (str): The ground truth text.
            model_response (str): The model's response text.
        Returns:
            float: The F1 score.
        """
        ground_truth_tokens = ground_truth.split()
        model_response_tokens = model_response.split()
        common = Counter(ground_truth_tokens) & Counter(model_response_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(model_response_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    async def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using LLM via agenerate.
        Args:
            text1 (str): First text
            text2 (str): Second text
        Returns:
            float: Similarity score between 0 and 1
        """
        template = """Rate the semantic similarity between these two texts on a scale of 0 to 1, where 1 means identical meaning and 0 means completely different:

Text 1: {text1}
Text 2: {text2}

Output only the numerical score between 0 and 1."""

        try:
            response = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={
                    "text1": text1,
                    "text2": text2
                },
                temperature=0.0,  # Use 0 temperature for consistent scoring
                output_parser=StrOutputParser(),
                structured_output=False
            )
            score = float(response.strip())
            return min(max(score, 0), 1)  # Clamp between 0 and 1
        except:
            return 0.0

    async def evaluate_belief_q(self, qa: dict[str, Any], model_response: str) -> tuple[bool, float]:
        """
        Evaluate the belief question using semantic similarity via LLM.
        """
        wrong_tom_view = qa['wrong_answer']
        correct_answer = qa['correct_answer']
        
        similarity_wrong = await self.compute_semantic_similarity(model_response, wrong_tom_view)
        similarity_correct = await self.compute_semantic_similarity(model_response, correct_answer)

        if similarity_wrong >= similarity_correct:
            wrong_view_lexical_overlap = self.compute_f1(wrong_tom_view, model_response)
            return False, wrong_view_lexical_overlap
        else:
            correct_view_lexical_overlap = self.compute_f1(correct_answer, model_response)
            return True, correct_view_lexical_overlap

    def evaluate_mc_belief_q(self, qa: dict[str, Any], model_response: str) -> bool:
        """
        Evaluate the multiple-choice version belief question.
        Args:
            qa (dict): The question and answer information.
            model_response (str): The model's response to the question.
        Returns:
            bool: True if the model's response matches the correct answer, False otherwise.
        """
        int_to_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        answer = int_to_alphabet[int(qa['correct_answer'])]
        response = model_response.lower()
        if response.startswith("(" + answer + ")") or response.startswith(answer + ")") or response.startswith(answer + "."): # a) or a. or a
            return True
        else:
            return False

    def evaluate_list_q_bracket(self, qa: dict[str, Any], model_response: str) -> tuple[bool, bool, bool, str]:
        """
        Check whether all the characters in the correct answer are in the model's response
        and none of the characters in the wrong answer are in the model's response
        Args:
            qa (dict): A dictionary containing the question and answer information.
            model_response (str): The response generated by the model.
        Returns:
            tuple: A tuple containing three values:
                - A boolean indicating whether the model's response satisfies the evaluation criteria.
                - A boolean indicating whether any aware characters were excluded from the model's response.
                - A boolean indicating whether any unaware characters were included in the model's response.
        """
        if model_response.count("[")>1:
            print(f"more than one brackets:\n{model_response}")
            answer_span = model_response
        elif model_response.count("[")==0 or model_response.count("]")==0:
            answer_span = model_response
        else:    
            answer_span = model_response[model_response.index("["):model_response.index("]")+1]
        excluded_aware_character = False
        included_unaware_character = False
        if type(qa['correct_answer'])==str:
            qa['correct_answer'] = str_to_list(qa['correct_answer'])
            qa['wrong_answer'] = str_to_list(qa['wrong_answer'])
        for character in qa['correct_answer']:
            if character.lower() not in answer_span.lower():
                excluded_aware_character = True
                break
        for character in qa['wrong_answer']:
            if character.lower() in answer_span.lower():
                included_unaware_character = True
                break
        return not(excluded_aware_character or included_unaware_character), excluded_aware_character, included_unaware_character, answer_span

    def map_binary_answer_to_int(self, model_response: str) -> int:
        """
        Maps a binary answer to an integer value.
        Args:
            model_response (str): The model's response.
        Returns:
            int: The mapped integer value. Returns 1 for positive answers (e.g., 'yes', 'true'), 
                 0 for negative answers (e.g., 'no', 'false'), and -1 for other cases.
        """
        model_answer = model_response.lower().strip("'").strip('"')
        if " yes," in model_answer or " yes " in model_answer or model_answer.startswith("yes") or " yes." in model_answer or " knows " in model_answer or model_answer.lower().startswith("true"):
            return 1
        elif " no," in model_answer or " no " in model_answer or model_answer.startswith("no") or " no." in model_answer or " does not know " in model_answer or " doesn't know " in model_answer or model_answer.lower().startswith("false"):
            return 0
        else:
            return -1

    def evaluate_binary_q_with_f1(self, qa: dict[str, Any], model_response: str) -> bool:
        """
        Evaluates a binary question with F1 score.
        Args:
            qa (dict): A dictionary containing the question and correct answer.
            model_response (str): The response generated by the model.
        Returns:
            bool: True if the model's response contains the correct answer, False otherwise.
        """
        tom_answer = qa['correct_answer'].split(":")[0] # for no:long
        model_answer = model_response.split()[0].lower().strip(",")
        if tom_answer in model_answer:
            return True
        else:
            return False

    def evaluate_fact_q(self, qa: dict[str, Any], model_response: str) -> float:
        result = self.compute_f1(qa['correct_answer'].lower(), model_response.lower())
        return result

    def yesno_to_int(self, yesno_str: str) -> int:
        mapping = {'yes': 1, 'no': 0, 'no:long': 0, 'error': -1}
        return mapping[yesno_str]

    async def evaluate_response(self, qas: list[dict[str, Any]], predictions: list[str]) -> list[dict[str, Any]]:
        """
        Evaluates the model's response for a list of questions and predictions.
        Args:
            qas (list): List of question-answer pairs.
            predictions (list): List of model predictions.
        Returns:
            list: Updated list of question-answer pairs with evaluation results and predictions.
        """
        print("Running evaluation...")
        assert len(qas) == len(predictions), "Number of questions and model predictions should be the same."
        for qa, pred in tqdm(zip(qas, predictions), total=len(qas)):
            if qa['question_type'].startswith("tom:belief:"):
                if qa['question_type'].endswith(":multiple-choice"):
                    result = self.evaluate_mc_belief_q(qa, pred)
                else:
                    result, word_overlap = await self.evaluate_belief_q(qa, pred)
                    qa['word_overlap'] = word_overlap
            elif qa['question_type'].endswith(":list"):
                result, excluded_aware_character, included_unaware_character, answer_span = self.evaluate_list_q_bracket(qa, pred)
                qa['excluded_aware_character'] = excluded_aware_character
                qa['included_unaware_character'] = included_unaware_character
                qa['prediction_answer_span'] = answer_span
            elif qa['question_type'].endswith(":binary"):
                _binary_answer = self.map_binary_answer_to_int(pred)
                if self.yesno_to_int(qa['correct_answer']) == _binary_answer:
                    result = True
                else:
                    result = False
                qa['binarized_model_answer'] = _binary_answer
            elif qa['question_type'].startswith("fact"):
                result = self.evaluate_fact_q(qa, pred) # type: ignore
            else:
                raise NotImplementedError
            qa['result'] = result
            qa['prediction'] = pred
        breakpoint()
        return qas

    def score_and_analyze(self, df: pd.DataFrame, target_scenario: str = 'inaccessible') -> dict[str, Any]:
        """
        Aggregates scores and performs analysis on the model responses and evaluation results.
        Args:
            df (pandas.DataFrame): The dataframe containing evaluation results
            target_scenario (str): Either 'inaccessible' or 'accessible'
        Returns:
            dict: Report containing various metrics and analyses
        """
        report = {}
        
        # Custom F1 calculation for binary questions
        def calculate_f1(predictions: list[int], references: list[int], pos_label: int = 0) -> float:
            tp = sum((p == pos_label and r == pos_label) for p, r in zip(predictions, references))
            fp = sum((p == pos_label and r != pos_label) for p, r in zip(predictions, references))
            fn = sum((p != pos_label and r == pos_label) for p, r in zip(predictions, references))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return f1

        # Convert string results to boolean/float if needed
        if type(df.result.iloc[0]) == str:
            df['result'] = df['result'].map(lambda x: x=='True' if x.endswith('e') else float(x))

        # Filter ToM questions and target scenario
        tom_df = df[df['question_type'].str.startswith("tom")].copy()
        target_df = tom_df[tom_df['missed_info_accessibility'] == target_scenario].copy()
        
        # Handle accessible scenario special case
        if target_scenario == 'accessible':
            _target_df = tom_df[tom_df['missed_info_accessibility'] == target_scenario].copy()
            set_ids = _target_df['set_id'].unique()
            target_sets = []
            for set_id in set_ids:
                if tom_df[tom_df['set_id'] == set_id]['missed_info_accessibility'].eq(target_scenario).all():
                    target_sets.append(set_id)
        else:
            target_sets = target_df['set_id'].unique() # type: ignore

        # ALL score calculations
        df1 = target_df[target_df['set_id'].isin(target_sets)].groupby("set_id")['result'].all()
        report[target_scenario+':set:ALL*'] = [df1.mean(), len(df1)]
        report[target_scenario+':set:ALL'] = [df1.mean(), len(df1)]

        # Belief Questions (multiple-choice)
        df1 = target_df[target_df['question_type'].str.endswith(":multiple-choice")]['result']
        report[target_scenario+':belief:multiple-choice'] = [df1.mean(), len(df1)]

        # Answerability Questions
        df1 = target_df[target_df['question_type'].str.startswith("tom:answerability")].groupby("set_id")['result'].all()
        report[target_scenario+':answerability:set:ALL'] = [df1.mean(), len(df1)]
        
        df1 = target_df[target_df['question_type'] == "tom:answerability:list"]['result']
        report[target_scenario+':answerability:list'] = [df1.mean(), len(df1)]
        
        if 'binarized_model_answer' in target_df.columns:
            answerability_responses = target_df[target_df['question_type'] == 'tom:answerability:binary']['binarized_model_answer'].tolist()
            answerability_refs = target_df[target_df['question_type'] == 'tom:answerability:binary']['correct_answer'].map(self.yesno_to_int).tolist()
            report[target_scenario+':answerability:binary-f1'] = [
                calculate_f1(answerability_responses, answerability_refs, pos_label=0),
                len(answerability_responses)
            ]

        # Info Accessibility Questions
        df1 = target_df[target_df['question_type'].str.startswith("tom:info_accessibility")].groupby("set_id")['result'].all()
        report[target_scenario+':info_accessibility:set:ALL'] = [df1.mean(), len(df1)]
        
        df1 = target_df[target_df['question_type']=="tom:info_accessibility:list"]['result']
        report[target_scenario+':info_accessibility:list'] = [df1.mean(), len(df1)]
        
        if 'binarized_model_answer' in target_df.columns:
            accessibility_responses = target_df[target_df['question_type'] == 'tom:info_accessibility:binary']['binarized_model_answer'].tolist()
            accessibility_refs = target_df[target_df['question_type'] == 'tom:info_accessibility:binary']['correct_answer'].map(self.yesno_to_int).tolist()
            report[target_scenario+':info_accessibility:binary-f1'] = [
                calculate_f1(accessibility_responses, accessibility_refs, pos_label=0),
                len(accessibility_responses)
            ]

        # Fact Questions
        df1 = df[df['question_type'].str.startswith("fact")]['result']
        report['fact_word-f1'] = [df1.mean(), len(df1)]

        # Error Analysis for List Questions
        if "tom:answerability:list" in target_df['question_type'].unique() and 'excluded_aware_character' in target_df.columns:
            list_wrong = target_df[
                (target_df['question_type']=="tom:answerability:list") & 
                (target_df['result'] == False)
            ][['excluded_aware_character', 'included_unaware_character']].copy()
            
            list_wrong['both'] = list_wrong['excluded_aware_character'] & list_wrong['included_unaware_character']
            list_wrong['reason'] = list_wrong.apply(
                lambda x: 'did_both' if x['both'] 
                else 'excluded_aware_character' if x['excluded_aware_character'] 
                else 'included_unaware_character', 
                axis=1
            )
            report[target_scenario+':tom:lists:wrong_reasons:freq'] = list_wrong['reason'].value_counts(normalize=False).to_dict() # type: ignore
        # Error Analysis for Binary Questions
        if 'binarized_model_answer' in target_df.columns:
            binary_wrong = target_df[
                (target_df['question_type'].str.endswith(":binary")) & 
                (target_df['result'] == False)
            ]['binarized_model_answer'].value_counts(normalize=False).to_dict()
            
            # Map error types
            error_mapping = {
                0: 'false_negative',
                1: 'false_positive',
                -1: 'irrelevant_response'
            }
            binary_wrong = {error_mapping.get(k, k): v for k, v in binary_wrong.items()}
            report[target_scenario+':tom:binary:wrong_reasons:freq'] = binary_wrong # type: ignore

        # Analysis by ToM order type
        if "tom:belief:inaccessible:multiple-choice" in tom_df.question_type.unique():
            belief_df = tom_df[
                tom_df['question_type'] == f'tom:belief:{target_scenario}:multiple-choice'
            ].copy()
            
            belief_df['tom_order'] = belief_df['tom_type'].map(lambda x: x.split(":")[0])
            df1 = belief_df.groupby('tom_order')['result'] # type: ignore
            tom_order_results = df1.value_counts(normalize=True)
            tom_order_counts = df1.value_counts()
            
            for idx in tom_order_results.index:
                if idx[1] == True:
                    report[f"{target_scenario}:{idx[0]}"] = [
                        tom_order_results[idx], 
                        int(tom_order_counts[idx[0]].sum())
                    ]

            # Cyclic vs Acyclic analysis
            df1 = belief_df.groupby('tom_type')['result'] # type: ignore
            belief_results = df1.value_counts(normalize=True)
            belief_counts = df1.value_counts()
            
            for idx in belief_results.index:
                if idx[1] == True:
                    report[f"{target_scenario}:{idx[0]}"] = [
                        belief_results[idx],
                        int(belief_counts[idx[0]].sum())
                    ]

        # # Character tracking analysis
        # binary_qas = df[df['question_type'].str.endswith(":binary")].copy()
        # binary_qas['target_character'] = binary_qas['question'].map(lambda x: x.removeprefix("Does ").split(" know")[0].lower())
        
        # belief_qas = target_df[target_df['question_type'].str.startswith("tom:belief")].copy()
        # belief_qas['target_character'] = belief_qas['question'].map(lambda x: x.lower().split("does ")[1].split()[0].lower())
        
        # answerability_list_qas = target_df[target_df['question_type'].str.endswith("answerability:list")].set_index("set_id", drop=False)
        # accessibility_list_qas = target_df[target_df['question_type'].str.endswith("info_accessibility:list")].set_index("set_id", drop=False)

        # # Analyze list question responses at character level
        # binary_answerability = binary_qas[binary_qas['question_type'].str.startswith('tom:answerability:')]
        # tiled_answerability = binary_answerability[["set_id", 'target_character', 'correct_answer']].join(
        #     answerability_list_qas[['prediction', "set_id"]], 
        #     on="set_id", 
        #     how='outer', 
        #     lsuffix='-binary'
        # )
        
        # tiled_answerability['binarized_model_answer'] = tiled_answerability.apply(
        #     lambda x: str(x['target_character']).lower() in str(x['prediction']).lower(), 
        #     axis=1
        # )
        # tiled_answerability['binarized_correct_answer'] = tiled_answerability['correct_answer'].map(
        #     lambda x: True if x =='yes' else False
        # )
        # tiled_answerability['result'] = tiled_answerability.apply(
        #     lambda x: x['binarized_model_answer'] == x['binarized_correct_answer'], 
        #     axis=1
        # )

        # binary_accessibility = binary_qas[binary_qas['question_type'].str.startswith('tom:info_accessibility:')]
        # tiled_accessibility = binary_accessibility[["set_id", 'target_character', 'correct_answer']].join(
        #     accessibility_list_qas[['prediction', "set_id"]], 
        #     on="set_id", 
        #     how='outer', 
        #     lsuffix='-binary'
        # )
        
        # tiled_accessibility['binarized_model_answer'] = tiled_accessibility.apply(
        #     lambda x: str(x['target_character']).lower() in str(x['prediction']).lower(), 
        #     axis=1
        # )
        # tiled_accessibility['binarized_correct_answer'] = tiled_accessibility['correct_answer'].map(
        #     lambda x: True if x =='yes' else False
        # )
        # tiled_accessibility['result'] = tiled_accessibility.apply(
        #     lambda x: x['binarized_model_answer'] == x['binarized_correct_answer'], 
        #     axis=1
        # )

        # # Calculate character-level metrics
        # df_for_all_character = pd.concat([
        #     belief_qas[['target_character', "set_id", 'result']],
        #     tiled_answerability[['target_character', "set_id", 'result']],
        #     tiled_accessibility[['target_character', "set_id", 'result']]
        # ])
        
        # df1 = df_for_all_character.groupby(["set_id", 'target_character'])['result'].all()
        # report[target_scenario+':set:ALL_character'] = [df1.mean(), len(df1)]

        # # Character consistency analysis
        # df_for_character_consistency = pd.concat([
        #     tiled_answerability[['target_character', "set_id", 'binarized_model_answer']],
        #     tiled_accessibility[['target_character', "set_id", 'binarized_model_answer']]
        # ])
        
        # df1 = df_for_character_consistency.reset_index(drop=True).groupby(
        #     ["set_id", 'target_character']
        # )['binarized_model_answer'].nunique().eq(1)
        
        # report[target_scenario+':set:character_answer_consistency'] = [df1.mean(), len(df1)]

        return report

async def fantom_simulation(row: pd.Series, engine: Optional[SocialWorldModel] = None) -> dict[str, Any]:  # type: ignore
    """Run experiment in simulation mode for FANToM benchmark (using ToM engine for memory tracking).
    
    Args:
        row: A pandas Series containing the question and other metadata
        engine: The ToM engine instance to use for simulation
        
    Returns:
        A dictionary containing the simulation results
    """
    assert engine is not None, "Engine must be provided"
    # Get the socialized context from the engine
    socialized_context = engine.existing_socialized_contexts[str(row['set_id'])]
    
    # Extract character information from the question
    question = row['complete_question']
    question_type = row['question_type']
    
    # Set appropriate agent prompt based on question type
    if question_type.startswith("tom:belief:"):
        engine.set_agent_prompt(
            "You will be asked a question about your beliefs based on a conversation you participated in. "
            "Be faithful to your memory of the conversation and do not hallucinate."
            "If some details are not provided in your memory, you are unlikely to know it unless if could be inferred from your knowledge of the conversation."
            "First reason about the question and then respond with the following format: "
            "<reasoning>(your step-by-step reasoning)</reasoning> <answer>(a or b)</answer>"
        )
    elif question_type.endswith(":list") or question_type.endswith(":binary"):
        engine.set_agent_prompt(
            "First reason and then respond with the following format: "
            "<reasoning>(your step-by-step reasoning)</reasoning> <answer>yes or no</answer>"
        )
    else:
        engine.set_agent_prompt(
            "First reason about the question and then respond with the following format: "
            "<reasoning>(your step-by-step reasoning)</reasoning> <answer>(your final answer)</answer>"
        )
    
    # Initialize the simulation with the socialized context
    await engine.initialize_simulation_from_socialized_context(socialized_context)
    
    # Extract target agent if present in the question
    target_agent = None
    if question_type.endswith(":binary"):
        # Extract target agent from questions like "Does Owen know..."
        _, question_content = question.split("\n")[:2]
        if question_content.startswith("Question: Does"):
            target_agent = question_content.removeprefix("Question: Does ").split(" know")[0].strip()
    elif question_type.endswith(":multiple-choice"):
        target_agent = question.replace("Question: What does ", "").split(" ")[0].strip()
    
    # Get agent names from the socialized context
    agent_names = socialized_context.agents_names
    # Get reasoning and answer from the engine
    if question_type.endswith(":list"):
        # For list questions, we need to ask each agent individually and gather their responses
        assert "List all the characters who know" in question, "List question must contain 'List all the characters who know'"
        information_asked_about = question.split("\n")[0].strip()
        if question.startswith("Information:"):
            question_for_each_agent = f"Do you know the following information: '{information_asked_about}'? (only answer with yes or no; you should answer yes if you know the information before the dialogue happens as well.)"
        else:
            question_for_each_agent = f"Do you know the precise correct answer to '{information_asked_about}'? (only answer with yes or no; you should answer yes if you know the answer before the dialogue happens as well.)"
        
        # Ask each agent individually using reason_about_belief
        # Initialize lists to store results
        all_reasoning = []
        knowledgeable_agents = []
        all_questions = []
        
        # Create tasks for all agents
        tasks = []
        for agent_name in agent_names:
            tasks.append(
                engine.reason_about_belief(
                    question_for_each_agent,
                    agent_names,
                    target_agent=agent_name,
                    answer_candidates=None
                )
            )
            all_questions.append(question_for_each_agent)
        # Gather results from all agents concurrently
        agent_results = await asyncio.gather(*tasks)
        
        # Process results
        for agent_name, (agent_reasoning, agent_answer) in zip(agent_names, agent_results):
            # Add agent's reasoning to the combined reasoning
            all_reasoning.append(f"{agent_name}'s reasoning: {agent_reasoning}")
            all_reasoning.append(f"{agent_name}'s answer: {agent_answer}")
            
            # Check if the agent knows the answer
            answer_lower = agent_answer.lower()
            if (answer_lower.startswith("yes") or 
                "yes, " in answer_lower or 
                " yes" in answer_lower or 
                answer_lower == "yes" or
                "i know" in answer_lower) and not (
                "no" in answer_lower.split()[0] or
                "don't know" in answer_lower or
                "do not know" in answer_lower or
                "doesn't know" in answer_lower or
                "does not know" in answer_lower
            ):
                knowledgeable_agents.append(agent_name)
        
        # Combine all reasoning
        combined_reasoning = "\n\n".join(all_reasoning)
        
        # Add a summary of the reasoning process
        summary = f"Based on individual questioning of each agent, the following agents indicated they know the answer to '{information_asked_about}': {', '.join(knowledgeable_agents) if knowledgeable_agents else 'None'}"
        reasoning = summary + "\n\n" + combined_reasoning
        
        # Format the answer as a list string
        answer = f"[{', '.join(knowledgeable_agents)}]"
        
        # Update simulation state
        engine.simulation.reasoning = reasoning
        engine.simulation.answer = answer
        engine.simulation.question = "\n".join(all_questions)
    else:
        # Use the existing method for other question types
        if target_agent in agent_names:
            reasoning, answer = await engine.reason_about_belief(
                question, 
                agent_names, 
                target_agent=target_agent,
            )
        else:
            reasoning, answer = "the agent is not in the world model", "no"
    # Get the simulation state
    simulation = engine.get_simulation()
    simulation_dict = simulation.dict()
    # Prepare the result
    result = {
        "question": question,
        "question_type": question_type,
        "reasoning": reasoning,
        "answer": answer,
        "correct_answer": row['correct_answer'],
        "socialized_context": socialized_context,
        "transformed_question": simulation_dict["question"],
        "memory": simulation_dict["agent_memories"],
        "target_agent": target_agent,
        "agents": simulation_dict["agents"]
    }
    return result