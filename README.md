# EveNet

- Option
    - EventInfo
        - MultiProcess.yaml
            - Model:
                - Normalization.pkl (from training dataset)
                - Balance.pkl (from training dataset)
            - Dataset

```yaml
INPUTS:
  point_cloud: (num_events, num_particles, num_features)
  point_cloud_mask: (num_events, num_particles)
  condition: (num_events, num_conditions)
  condition_mask: (num_events, 1) # ?? should pad?

  classification: (num_events, 1) # one-hot in model
  regression: (num_events, num_regressions)
  regression_mask: (num_events, num_regressions)

  # num_assignment = num_process * num_level1_particles
  assignment_indices: (num_events, num_assignment, num_level2_particles) # with padding
  assignment_indices_mask: (num_events, num_assignment, num_level2_particles) # with padding
  assignment_mask: (num_events, num_assignment)

  num_vectors: (num_events, ) # num_particles
  num_sequential_vectors: (num_events, ) # num_particles
```



