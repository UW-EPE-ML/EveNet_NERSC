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


| Function                                 | Call Count | Total Time [sec] | Average Time [sec] |
|------------------------------------------|------------|------------------|---------------------|
| EveNetEngine.backward                    | 5          | 1.7846           | 0.3569              |
| EveNetEngine.on_train_epoch_end          | 1          | 0.0002           | 0.0002              |
| EveNetEngine.on_validation_epoch_end     | 1          | 12.5136          | 12.5136             |
| EveNetEngine.shared_step                 | 10         | 3.6951           | 0.3695              |
| EveNetEngine.training_step               | 5          | 3.6671           | 0.7334              |
| [Assignment] assignment_cross_entropy_loss | 280        | 0.0112           | 0.0000401           |
| [Assignment] convert_target_assignment   | 10         | 0.0009           | 0.0000902           |
| [Assignment] convert_target_assignment_array | 10     | 0.0021           | 0.0002146           |
| [Assignment] loss_single_process         | 160        | 0.0628           | 0.0003923           |
| [Assignment] predict                     | 160        | 0.5675           | 0.003547            |
| [Assignment] reconstruct_mass_peak       | 856        | 0.0399           | 0.0000466           |
| [Assignment] shared_epoch_end            | 1          | 9.9822           | 9.9822              |
| [Assignment] shared_step                 | 10         | 0.7446           | 0.0745              |
| [Classification] shared_epoch_end        | 1          | 2.5312           | 2.5312              |
| [Classification] shared_step             | 10         | 0.0108           | 0.001081            |



