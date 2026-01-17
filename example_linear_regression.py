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

print(f"\nLearned regression model:Y = {model.coef_[0]:.3f} * X + {model.intercept_:.3f}")

#previsão para novos dados
X_new = np.array([[12], [14], [16]])
y_pred_new = model.predict(X_new)

#gráfico dos resultados
plt.figure(figsize=(8, 5))
plt.gca().set_axisbelow(True)
plt.scatter(X, Y, label="Training data")
X_line = np.linspace(0, 16, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, label="Regression Line")
plt.scatter(X_new, y_pred_new, color="deeppink", s=70, label="New predictions", zorder=3)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Example")
plt.legend()
plt.grid(True)
plt.savefig("example_linear_regression.png", dpi=150)
plt.close()

#output novas previsões
print("\nPredictions for new input values:")
for i in range(len(X_new)):
    print(f"  For X = {X_new[i, 0]:.1f}, predicted Y = {y_pred_new[i]:.2f}")
    
#output error inicial vs final
initial_loss = model.loss_history_[0]
final_loss = model.loss_history_[-1]
print("\nTraining error summary:")
print(f"  Initial MSE: {initial_loss:.4f}")
print(f"  Final MSE:   {final_loss:.4f}")


#output iterações + razão de paragem
print()
print(f"Iterations: {model.n_iter_}")
print(f"Stop reason: {model.stop_reason_}")
    
#graf da evolução do erro (loss)
plt.figure(figsize=(7, 4))
plt.plot(model.loss_history_)
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Training Error Evolution")
plt.yscale("log")  
plt.grid(True)
plt.savefig("example_loss_curve.png", dpi=150)
plt.close()

