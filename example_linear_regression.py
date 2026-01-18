#Exemplo de Aplicação
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

#dados
np.random.seed(42)
X = np.linspace(0, 15, 42).reshape(-1, 1)
Y = 1.5 * X.flatten() + 4.5 + np.random.normal(0, 0.5, size=42)

#treino do modelo
model = LinearRegression(
    learning_rate=0.005,
    n_iterations=3000,
    tol=1e-6, 
    min_loss=2
)

model.fit(X, Y)

#previsão para novos dados
X_new = np.array([[12], [14], [16]])
y_pred_new = model.predict(X_new)

#output novas previsões
print("\nPredictions for new input values:")
for i in range(len(X_new)):
    print(f"  For X = {X_new[i, 0]:.1f}, predicted Y = {y_pred_new[i]:.2f}")
    
#output iterações + razão de paragem
print()
print(f"Iterations: {model.n_iter_}")
print(f"Stop reason: {model.stop_reason_}")

#gráfico linear regression
r2 = model.score(X, Y) #calcular erro

plt.figure(figsize=(9, 5))
plt.gca().set_axisbelow(True)
plt.scatter(X, Y, color="darkorange", label="Training data")

X_line = np.linspace(0, 16, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, label="Regression Line")
plt.scatter(X_new, y_pred_new, color="deeppink", s=70, label="New predictions", zorder=3)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Example")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.grid(True)

info_text = (
    f"y = {model.coef_[0]:.2f}·X + {model.intercept_:.2f}\n"
    f"$R^2$ = {r2:.3f}")

plt.text(
    1.06, 0.78,
    info_text,
    transform=plt.gca().transAxes,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round", edgecolor="deeppink", facecolor="white"))

plt.subplots_adjust(right=0.75)
plt.savefig("example_linear_regression.png", dpi=150)
plt.close()
    
#graf da evolução do erro (loss)
initial_loss = model.loss_history_[0]
final_loss = model.loss_history_[-1]

plt.figure(figsize=(7, 4))
plt.plot(model.loss_history_, color="deeppink")
plt.yscale("log")

mse_text = (
    f"Initial MSE: {initial_loss:.4f}\n"
    f"Final MSE: {final_loss:.4f}")

plt.text(
    0.01, 1.10,
    mse_text,
    transform=plt.gca().transAxes,
    ha="left",
    bbox=dict(boxstyle="round", edgecolor="deeppink", facecolor="white"))

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Training Error Evolution")
plt.grid(True)
plt.subplots_adjust(top=0.80)
plt.savefig("example_loss_curve.png", dpi=150)
plt.close()
