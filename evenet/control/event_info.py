from collections import OrderedDict
from itertools import chain, permutations
from functools import cache
import re

import torch
from evenet.utilities.group_theory import complete_indices, symmetry_group

from evenet.dataset.types import *
from evenet.utilities.group_theory import (
    power_set,
    complete_symbolic_symmetry_group,
    complete_symmetry_group,
    expand_permutations
)


def cached_property(func):
    return property(cache(func))


def with_default(value, default):
    return default if value is None else value


def key_with_default(database, key, default):
    if key not in database:
        return default

    value = database[key]
    return default if value is None else value


class EventInfo:
    def __init__(
            self,

            # Information about observable inputs for this event.
            input_types: InputDict[str, InputType],
            input_features: InputDict[str, Tuple[FeatureInfo, ...]],

            # Information about the target structure for this event.
            event_particles: Dict[str, Particles],
            product_particles: Dict[str, EventDict[str, Particles]],

            # Information about auxiliary values attached to this event.
            regressions: FeynmanDict[str, List[RegressionInfo]],
            classifications: FeynmanDict[str, List[ClassificationInfo]],
            class_label: Dict[str, Dict],
            resonance_info: Dict[str, Dict],
            resonance_particle_properties: List
    ):

        self.input_types = input_types
        self.input_names = list(input_types.keys())
        self.input_features = input_features

        self.event_particles = event_particles
        self.event_mapping = OrderedDict()
        self.event_symmetries = OrderedDict()

        self.product_particles = product_particles
        self.product_mappings: ODict[str, ODict[str, ODict[str, int]]] = OrderedDict()
        self.product_symmetries: ODict[str, ODict[str, Symmetries]] = OrderedDict()

        self.process_names = list(self.event_particles.keys())
        self.resonance_info = resonance_info
        self.resonance_particle_properties = resonance_particle_properties


        for process in self.event_particles:

            self.event_mapping[process] = self.construct_mapping(self.event_particles[process])
            self.event_symmetries[process] = Symmetries(
                len(self.event_particles[process]),
                self.apply_mapping(self.event_particles[process].permutations, self.event_mapping[process])
            )

            product_mappings_process = OrderedDict()
            product_symmetries_process = OrderedDict()

            for event_particle, product_particles in self.product_particles[process].items():
                product_mapping = self.construct_mapping(product_particles)

                product_mappings_process[event_particle] = product_mapping
                product_symmetries_process[event_particle] = Symmetries(
                    len(product_particles),
                    self.apply_mapping(product_particles.permutations, product_mapping)
                )
            self.product_mappings[process] = product_mappings_process
            self.product_symmetries[process] = product_symmetries_process

        self.event_permutation = OrderedDict()
        self.max_event_particles = 1
        for process_name in self.process_names:
            event_permutation = complete_indices(
                self.event_symmetries[process_name].degree,
                self.event_symmetries[process_name].permutations
            )
            self.event_permutation[process_name] = event_permutation
            for permutation_group in event_permutation:
                for permutation_element in permutation_group:
                    max_indices = len(list(permutation_element))
                    if max_indices > self.max_event_particles:
                        self.max_event_particles = max_indices


        self.regressions = regressions
        self.regression_types = {"/".join([SpecialKey.Event] + [target.name]): target.type for target in regressions[SpecialKey.Event]}
        for process in self.product_particles:
            if process in regressions:
                for particle in regressions[process]:
                    if isinstance(regressions[process][particle], dict):
                        for product in regressions[process][particle]:
                            for target in regressions[process][particle][product]:
                                key = "/".join([process, particle, product] + [target.name])
                                self.regression_types[key] =  target.type
                    else:
                        for target in regressions[process][particle]:
                            key = "/".join([process, particle] + [target.name])
                            self.regression_types[key] = target.type

        self.regression_names     = self.regression_types.keys()
        self.num_regressions = len(self.regression_names)


        self.classifications = classifications
        self.classification_names = ['/'.join([SpecialKey.Event, target]) for target in self.classifications[SpecialKey.Event]]
        self.class_label = class_label
        self.num_classes = dict()
        for name in class_label['EVENT']:
            self.num_classes[name] = (np.array(class_label['EVENT'][name])).shape[-1]

        self.pairing_topology = OrderedDict()
        self.pairing_topology_category = OrderedDict()

        resonance_particle_properties_summary = []
        for process in self.process_names:
            for event_particle_name, product_symmetry in self.product_symmetries[process].items():
                topology_name = ''.join(self.product_particles[process][event_particle_name].names)
                topology_name = f"{event_particle_name}/{topology_name}"
                topology_name = re.sub(r'\d+', '', topology_name)

                pairing_topology_category = topology_name
                resonance_particle_properties_tmp = []
                for name, sub_dict in self.resonance_info.items():
                    for sub_name, third_dict in sub_dict.items():
                        if topology_name == sub_name:
                            pairing_topology_category = name
                            for property_name in self.resonance_particle_properties:
                                if property_name in third_dict:
                                    resonance_particle_properties_tmp.append(third_dict[property_name])
                                #                                print(f"----{property_name}: {third_dict[property_name]}")
                                else:
                                    resonance_particle_properties_tmp.append(0.0)
                #                                print(f"----{property_name}: 0.0 (no entry so pad zero)")

                resonance_particle_properties_summary.append(np.array(resonance_particle_properties_tmp))
                if topology_name not in self.pairing_topology:
                    self.pairing_topology[topology_name] = {
                        "product_particles": self.product_particles[process][event_particle_name],
                        "product_symmetry": product_symmetry,
                        "pairing_topology_category": pairing_topology_category,
                        "resonance_particle_properties": torch.Tensor(np.array(resonance_particle_properties_tmp))}
                    if pairing_topology_category not in self.pairing_topology_category:
                        self.pairing_topology_category[pairing_topology_category] = {
                            "product_particles": self.product_particles[process][event_particle_name],
                            "product_symmetry": product_symmetry,
                            "nCond": len(resonance_particle_properties_tmp)}

        self.assignment_names = OrderedDict()
        for process in self.product_particles:
            self.assignment_names[process] = []
            for event_particle, daughter_particles in self.product_particles[process].items():
                self.assignment_names[process].append(event_particle)

        if (len(resonance_particle_properties_summary) == 0):
            self.resonance_particle_properties_mean = np.array([0.0])
            self.resonance_particle_properties_std = np.array([1.0])
        else:
            resonance_particle_properties_summary = np.stack(resonance_particle_properties_summary)
            self.resonance_particle_properties_mean = np.mean(resonance_particle_properties_summary, axis=0)
            self.resonance_particle_properties_std = np.std(resonance_particle_properties_summary, axis=0)
            self.resonance_particle_properties_std[self.resonance_particle_properties_std < 1e-6] = 1.0

        self.resonance_particle_properties_mean = torch.Tensor(self.resonance_particle_properties_mean)
        self.resonance_particle_properties_std = torch.Tensor(self.resonance_particle_properties_std)

    def normalized_features(self, input_name: str) -> NDArray[bool]:
        return np.array([feature.normalize for feature in self.input_features[input_name]])

    def log_features(self, input_name: str) -> NDArray[bool]:
        return np.array([feature.log_scale for feature in self.input_features[input_name]])

    @cached_property
    def event_symbolic_group(self) -> ODict[str, SymbolicPermutationGroup]:
        event_symbolic_group_dict = OrderedDict()
        for process in self.process_names:
            event_symbolic_group_dict[process] = complete_symbolic_symmetry_group(*(self.event_symmetries[process]))
        return event_symbolic_group_dict

    @cached_property
    def event_permutation_group(self) -> ODict[str, PermutationGroup]:
        event_permutation_group_dict = OrderedDict()
        for process in self.process_names:
            event_permutation_group_dict[process] = complete_symmetry_group(*(self.event_symmetries[process]))
        return event_permutation_group_dict

    @cached_property
    def ordered_event_transpositions(self) -> ODict[str, Set[List[int]]]:
        ordered_event_transpositions_dict = OrderedDict()
        for process in self.event_symbolic_group:
            ordered_event_transpositions_dict[process] = set(chain.from_iterable(
                e.transpositions()
                for e in self.event_symbolic_group[process].elements
            ))
        return ordered_event_transpositions_dict

    @cached_property
    def event_transpositions(self) -> ODict[str, Set[Tuple[int, int]]]:
        event_transpositions_dict = OrderedDict()
        for process in self.ordered_event_transpositions:
            event_transpositions_dict[process] = set(
                map(tuple, map(sorted, self.ordered_event_transpositions[process])))
        return event_transpositions_dict

    @cached_property
    def event_equivalence_classes(self) -> ODict[str, Set[FrozenSet[FrozenSet[int]]]]:

        event_equivalence_classes_dict = OrderedDict()
        for process in self.event_symmetries:
            num_particles = self.event_symmetries[process].degree
            group = self.event_symbolic_group[process]
            sets = map(frozenset, power_set(range(num_particles)))
            event_equivalence_classes_dict[process] = set(
                frozenset(frozenset(g(x) for x in s) for g in group.elements) for s in sets)
        return event_equivalence_classes_dict

    @cached_property
    def product_permutation_groups(self) -> ODict[str, ODict[str, PermutationGroup]]:

        product_permutation_groups_dict = OrderedDict()
        for process in self.product_symmetries:
            output = []
            for name, (degree, symmetries) in self.product_symmetries[process].items():
                symmetries = [] if symmetries is None else symmetries
                permutation_group = complete_symmetry_group(degree, symmetries)
                output.append((name, permutation_group))
            product_permutation_groups_dict[process] = OrderedDict(output)

        return product_permutation_groups_dict

    @cached_property
    def product_symbolic_groups(self) -> ODict[str, ODict[str, SymbolicPermutationGroup]]:

        product_symbolic_groups_dict = OrderedDict()
        for process in self.product_symmetries:
            output = []
            for name, (degree, symmetries) in self.product_symmetries[process].items():
                symmetries = [] if symmetries is None else symmetries
                permutation_group = complete_symbolic_symmetry_group(degree, symmetries)
                output.append((name, permutation_group))
            product_symbolic_groups_dict[process] = OrderedDict(output)
        return product_symbolic_groups_dict

    def num_features(self, input_name: str) -> int:
        return len(self.input_features[input_name])

    def input_type(self, input_name: str) -> InputType:
        return self.input_types[input_name].upper()

    @staticmethod
    def parse_list(list_string: str):
        return tuple(map(str.strip, list_string.strip("][").strip(")(").split(",")))

    @staticmethod
    def construct_mapping(variables: Iterable[str]) -> ODict[str, int]:
        return OrderedDict(map(reversed, enumerate(variables)))

    @staticmethod
    def apply_mapping(permutations: Permutations, mapping: Dict[str, int]) -> MappedPermutations:
        return [
            [
                tuple(
                    mapping[element]
                    for element in cycle
                )
                for cycle in permutation
            ]
            for permutation in permutations
        ]

    @classmethod
    def construct(cls, config: dict, resonance_info: dict):
        # Extract input feature information.
        # ----------------------------------
        input_types = OrderedDict()
        input_features = OrderedDict()

        for input_type in config[SpecialKey.Inputs]:
            current_inputs = with_default(config[SpecialKey.Inputs][input_type], default={})

            for input_name, input_information in current_inputs.items():
                input_types[input_name] = input_type.upper()
                input_features[input_name] = tuple(
                    FeatureInfo(
                        name=name,
                        normalize=("normalize" in normalize.lower()) or ("true" in normalize.lower()),
                        log_scale="log" in normalize.lower()
                    )

                    for name, normalize in input_information.items()
                )

        # Extract event and permutation information.
        # ------------------------------------------
        permutation_config = key_with_default(config, SpecialKey.Permutations, default={})

        event_particles_summary = OrderedDict()
        product_particles_summary = OrderedDict()

        event_particles = OrderedDict()  # Default value
        product_particles = OrderedDict()  # Default value

        for process in permutation_config:
            event_names = tuple(config[SpecialKey.Event][process].keys())
            event_permutations = key_with_default(permutation_config[process], SpecialKey.Event, default=[])
            event_permutations = expand_permutations(event_permutations)
            event_particles = Particles(event_names, event_permutations)
            product_particles = OrderedDict()

            for event_particle in event_particles:
                products = config[SpecialKey.Event][process][event_particle]

                product_names = [
                    next(iter(product.keys())) if isinstance(product, dict) else product
                    for product in products
                ]

                product_sources = [
                    next(iter(product.values())) if isinstance(product, dict) else None
                    for product in products
                ]

                input_names = list(input_types.keys())
                product_sources = [
                    input_names.index(source) if source is not None else -1
                    for source in product_sources
                ]

                product_permutations = key_with_default(permutation_config[process], event_particle, default=[])
                product_permutations = expand_permutations(product_permutations)

                product_particles[event_particle] = Particles(product_names, product_permutations, product_sources)

            event_particles_summary[process] = event_particles
            product_particles_summary[process] = product_particles

        # Extract Regression Information.
        # -------------------------------
        regressions = key_with_default(config, SpecialKey.Regressions, default={})
        regressions = feynman_fill(regressions, event_particles, product_particles, constructor=list)

        # Fill in any default parameters for regressions such as gaussian type.
        regressions = feynman_map(
            lambda raw_regressions: [
                RegressionInfo(*(regression if isinstance(regression, list) else [regression]))
                for regression in raw_regressions
            ],
            regressions
        )

        # Extract Classification Information.
        # -----------------------------------
        classifications = key_with_default(config, SpecialKey.Classifications, default={})
        classifications = feynman_fill(classifications, event_particles, product_particles, constructor=list)

        class_label = key_with_default(config, SpecialKey.ClassLabel, default={})

        resonance_particle_property = key_with_default(config, SpecialKey.ParticleProperties, default=[])

        # TODO: feynman_fill (not necessary, but would be nice)

        return cls(
            input_types,
            input_features,
            event_particles_summary,
            product_particles_summary,
            regressions,
            classifications,
            class_label,
            resonance_info,
            resonance_particle_property
        )
