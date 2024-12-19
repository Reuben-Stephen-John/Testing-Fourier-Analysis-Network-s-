import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(fan_model, gated_fan_model, mlp_model,device):
    x_test = torch.linspace(0, 10, 1000).reshape(-1, 1).float().to(device)
    y_test = np.sin(2 * np.pi * x_test.cpu().numpy()) + np.cos(3 * np.pi * x_test.cpu().numpy())
    
    fan_model.eval()
    mlp_model.eval()
    with torch.no_grad():
        fan_pred = fan_model(x_test).cpu().numpy()
        gated_fan_pred = gated_fan_model(x_test).cpu().numpy()
        mlp_pred = mlp_model(x_test).cpu().numpy()

    # Plotting the results
    plt.figure(figsize=(12, 6))

    plt.plot(x_test.cpu().numpy(), y_test, label="True Function", color="black", linestyle="dashed")
    plt.plot(x_test.cpu().numpy(), fan_pred, label="FAN Prediction", color="blue", alpha=0.7)
    plt.plot(x_test.cpu().numpy(), gated_fan_pred, label="GATED FAN Prediction", color="green", alpha=0.7)
    plt.plot(x_test.cpu().numpy(), mlp_pred, label="MLP Prediction", color="red", alpha=0.7)
    
    plt.legend()

    plt.title("Comparison of FAN and Gated FAN and MLP on Symbolic Dataset")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show(block=False)  # Blocks the script until the window is closed
    plt.savefig("evaluation_plot.png")  # Uncomment to save the plot as an image