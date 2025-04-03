export interface SocialContext {
  timestep: string;
  state: string;
  observations: string[];
  actions: string[];
}

export interface SocialData {
  agents_names: string[];
  socialized_context: SocialContext[];
}
