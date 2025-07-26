import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# สุ่มค่า m, a
np.random.seed(42)
n = 100
m = np.random.uniform(0, 40, n)    # มวล (0-40 kg)
a = np.random.uniform(0, 5, n)    # ความเร่ง (0 ถึง 10 m/s^2)

# คำนวณ F = ma
F = m * a

# DataFrame จากการสุ่มค่า m, a และคำนวณค่าหา F
df = pd.DataFrame({
    'mass': m,
    'acceleration': a,
    'force': F
})

df.head()

#plot กราฟที่ได้จากการคำนวณ
import seaborn as sns

sns.pairplot(df)
plt.show()

print(df.describe())

# Features = mass, acceleration
X = df[['mass', 'acceleration']]
y = df['force']  # Target variable

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

# ดูค่าพารามิเตอร์ของโมเดล
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# พยากรณ์ค่า F ที่โมเดลทำนายได้
y_pred = model.predict(X)

# ค่าความผิดพลาดของโมเดล
print("Mean Squared Error (MSE):", mean_squared_error(y, y_pred))
print("R-squared (R²) Score:", r2_score(y, y_pred))

# กราฟเปรียบเทียบค่าแรงจริงและแรงที่โมเดลทำนายได้
plt.scatter(y, y_pred, alpha=0.8)
plt.xlabel("Actual Force (F)")
plt.ylabel("Predicted Force")
plt.title("Actual vs Predicted Force")
plt.grid(True)
plt.show()
