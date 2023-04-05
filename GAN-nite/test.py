import os
from tkinter import Y
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from ganite import Ganite
from ganite.datasets import load
from ganite.utils.metrics import sqrt_PEHE_with_diff


X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("twins")

print(W_train.shape)
print(Y_train.shape)
model = Ganite(X_train, W_train, Y_train, num_iterations=500)

pred = model(X_test).numpy()

pehe = sqrt_PEHE_with_diff(Y_test, pred)
print(Y_test.shape)
print(pred.shape)

print(f"PEHE score for GANITE on MBQIP = {pehe}")
