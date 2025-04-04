export interface SocialContext {
  timestep: string;
  state: string;
  observations: Record<string, string>;
  actions: Record<string, string>;
}

export interface SocialData {
  agents_names: string[];
  socialized_context: SocialContext[];
}
