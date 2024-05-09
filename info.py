from pettingzoo.utils.env import ActionType, AgentID


class AgentInfo(object):
    def __init__(self, name: AgentID):
        self.name = name
        self.actions: list[ActionType] = []
        self.event_list: list[list[str]] = []
        self.properties_list: dict[str, list[float]] = {}
        self.current_step = 0
        self.episode = 0

    def reset(self):
        self.episode += 1
        self.actions = []
        self.event_list = []
        self.properties_list = {k: [] for k in self.properties_list.keys()}
        self.current_step = 0

    def update(self, action: ActionType, events: list[str], properties: dict[str, float]):
        self.actions.append(action)
        self.event_list.append(events)
        for agent_property, property_value in properties.items():
            self.properties_list.setdefault(agent_property, [0.0] * self.episode).append(property_value)
        self.current_step += 1

    def get_dict(self):
        if self.current_step == 0:
            d = {'actions': [], 'events': []}  #, 'episode': str(self.episode)
            d.update({k.capitalize(): [] for k, p_l in self.properties_list.items()})
            return d
        d = {'actions': self.actions[-1], 'events': self.event_list[-1]}  # , 'episode': str(self.episode)
        d.update({k.capitalize(): p_l for k, p_l in self.properties_list.items()})
        return d

    def info_current(self) -> str:
        if not self.actions:
            return (f"Agent: ; " +
                    f"Actions: ; " +
                    f"Events: ; " +
                    "; ".join(f"{k.capitalize()}: {p_l[-1]}" for k, p_l in self.properties_list.items())
                    )
        return (f"Agent: {self.name}; " +
                f"Actions: {self.actions[-1]}; " +
                f"Events: {self.event_list[-1]}; " +
                "; ".join(f"{k.capitalize()}: {p_l[-1]}" for k, p_l in self.properties_list.items())
                )

    def get_summery(self) -> tuple[dict[ActionType, int], dict[str, int], dict[str, float]]:
        actions = {}
        for a in self.actions:
            actions[a] = actions.get(a, 0) + 1
        events = {}
        for e_l in events:
            for e in e_l:
                events[e] = events.get(e, 0) + 1
        properties = {k: sum(v) for k, v in self.properties_list.items()}

        return actions, events, properties

    def info(self):
        actions, events, properties = self.get_summery()
        return (f"Name:{self.name}; "
                f"Actions: {actions}; " +
                f"Events: {events}; " +
                f"Properties: {properties}"
                )


class Info(object):
    def __init__(self, agent_list: list[AgentID]):
        self.dataset_times = 0
        self.agent_info_object = {agent: AgentInfo(agent) for agent in agent_list}

    def update_agent(self, agent: AgentID, action: ActionType, events: list[str], properties: dict[str, float]):
        self.agent_info_object[agent].update(action, events, properties)

    def reset(self):
        for ao in self.agent_info_object.values():
            ao.reset()

    def info_current(self):
        return {an: a.get_dict() for an, a in self.agent_info_object.items()}

    def info_agents(self):
        return "\n".join([a.info() for a in self.agent_info_object.values()])

    def get_dict(self):
        return {agent: ao.get_dict() for agent, ao in self.agent_info_object.items()}

    def info(self, normalize: bool = True):
        actions, events, properties = {}, {}, {}
        for agent, agent_info_o in self.agent_info_object.items():
            tmp = agent_info_o.get_summery()
            actions[agent], events[agent], properties[agent] = tmp[0], tmp[1], tmp[2]

        actions_s, events_s, properties_s = {}, {}, {}
        for agent in self.agent_info_object.keys():
            for a, v in actions[agent].items():
                if normalize:
                    v = v / len(self.agent_info_object)
                actions_s[a] = actions_s.get(a, 0) + v

            for e, v in events[agent].items():
                if normalize:
                    v = v / len(self.agent_info_object)
                events_s[e] = events_s.get(e, 0) + v

            for p, v in properties[agent].items():
                if normalize:
                    v = v / len(self.agent_info_object)
                properties_s[p] = properties_s.get(p, 0) + v

        return (f"Actions: {actions}; " +
                f"Events: {events}; " +
                f"Properties: {properties}"
                )
