from NewModel import DiscreteCenterPredictorCNN

model = DiscreteCenterPredictorCNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params} parameters")