import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# cost function & gradient
def f(x): return (x[0] - 3)**2 + (x[1]-5)**2 + 10
def grad(x): return 2 * (x - np.array([3, 5]))


# 학습율(lr) 및 멈춤 조건(pause)
iter_count = 0
lr = 0.2
pause = 0.01
x = np.array([10, 20])        # x 초기 설정값
x_vals = [x.tolist()]         # array 를 list 로 반환
cost_vals = [f(x)]
prev_cost = f(x)

print(x)
while True:
    iter_count += 1
    x = x - lr * grad(x)
    curr_cost = f(x)
    print("%3d-th iteration: x = [%0.4f, %0.4f]\
        cost = %0.4f" % (iter_count, x[0], x[1], f(x)))

    # 멈춤 조건 설정
    if curr_cost > prev_cost or np.abs(curr_cost - prev_cost) < pause:
        break

    x_vals.append(x.tolist())
    cost_vals.append(curr_cost)
    prev_cost = curr_cost

print("Final result : x = [%0.4f , %0.4f], cost = %0.4f \
    at iteration = %d\n" % (x[0], x[1], f(x), iter_count))


# 여러 개의 데이터를 다른 형태로 변환(map), 동인한 개수로 이루어진 자료형을 묶음(zip)
x1, x2 = map(list, zip(*x_vals))
ax = plt.axes(projection="3d")
ax.plot3D(x1, x2, cost_vals, "ro-")
plt.grid(axis="both")
plt.title("Example of SGD algorithm")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
plt.show()

plt.scatter(x1, x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(axis="both")
plt.show()
