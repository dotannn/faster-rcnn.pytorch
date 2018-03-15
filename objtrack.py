import numpy as np


class Instance:
    _instances_count = 0
    def __init__(self, event):
        self._features = event._features
        self._events = [event]
        self._stability = 1
        self._bbox = event._bbox
        self._label = event._label

        self._score = event._score
        self._com = event._com

        self._estimate_object_distance()

        self._name = "%s_%d" % (self._label, Instance._instances_count)
        Instance._instances_count += 1


    @property
    def stability(self):
        return self._stability

    @property
    def features(self):
        return self._features

    def update(self, ev, partial=False):
        self._events.append(ev)
        self._features = ev._features
        self._label = ev._label
        self._score = ev._score
        self._com = ev._com
        self._bbox = ev._bbox
        self._estimate_object_distance()
        self._stability = min(self._stability + 1, 10)


    def _estimate_object_distance(self, far_threshold=3.0, near_threshold=1.5):

        _person_width_in_meters = 0.75
        _focal_length = 450.0


        width_in_pix = float(self._bbox[3]) - float(self._bbox[1])

        dist = (_person_width_in_meters * _focal_length) / width_in_pix

        self._distance = dist

        if dist < near_threshold:
            self._distance_level = 'Near'
        elif dist > far_threshold:
            self._distance_level = 'Far'
        else:
            self._distance_level = 'Mid'


class ObjectsTracker:
    def __init__(self):
        self._instances = []
        self._appearance_factor = 0.5
        self._euclidean_factor = 0.0

        self._threshold = 45.0

    @property
    def instances(self):
        return self._instances


    @staticmethod
    def L2(a, b):
        return np.linalg.norm(a-b)


    def match_events_to_instances(self, events):
        if len(self._instances) == 0:
            return events

        total_dist = np.zeros((len(events), len(self._instances)))

        for i, inst in enumerate(self._instances):
            for j, ev in enumerate(events):
                total_dist[j, i] = self._appearance_factor * self.L2(ev._features, inst._features) + \
                                   self._euclidean_factor * self.L2(ev._com, inst._com) \
                                   + 1000 * float(ev._label != inst._label)

        unused_events = []
        matched_insts = []
        for j, ev in enumerate(events):
            i = np.argmin(total_dist[j, :])
            val = total_dist[j, i]
            if val < self._threshold:
                inst = self._instances[i]
                inst.update(ev)
                matched_insts.append(inst)
            else:
                unused_events.append(ev)

        for inst in self._instances:
            if inst in matched_insts:
                continue
            inst._stability -= 1

        return unused_events


    def initiate_new_instances(self, events):
        for event in events:
            self._instances.append(Instance(event))


    def filter_events(self, events):
        return events

    def track(self, events):
        events = self.filter_events(events)
        unmatched_events = self.match_events_to_instances(events)
        self.initiate_new_instances(unmatched_events)

        self.remove_old_instances()

        return self._instances

    def remove_old_instances(self):
        keep_instances = []
        for inst in self._instances:
            if inst.stability > 0:
                keep_instances.append(inst)


        self._instances = keep_instances

    def current_proposals(self):
        return []
