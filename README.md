# EveNet


- Option
  - EventInfo 
    - MultiProcess.yaml
      - Model:
        - Normalization.pkl (from training dataset)
        - Balance.pkl (from training dataset)
      - Dataset

```python
options

eventInfo = EventInfo()

data = pd.read_csv('data.csv')
data = preprocess(eventInfo)

model = load_model('model.h5', eventInfo)

model(data)

```

- Global_Control (class):
  - Options
  - EventInfo

```yaml
# Global_Control.yaml

train:
  layer: 4
  trainable: true

include: 
  - train.yaml
  - eval.v1.yaml
  - eval.v2.yaml
```