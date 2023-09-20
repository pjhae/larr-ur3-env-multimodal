
import numpy as np

def generate_action_sequence(start_point, end_point, max_distance):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    num_steps_x = int(abs(dx) / max_distance[0])  # x축 이동 횟수
    num_steps_y = int(abs(dy) / max_distance[1])  # y축 이동 횟수
    num_steps = max(num_steps_x, num_steps_y)  # 더 큰 이동 횟수를 선택

    action_sequence = []

    if num_steps > 0:
        step_x = dx / num_steps
        step_y = dy / num_steps

        for _ in range(num_steps):
            action_sequence.append((step_x, step_y))

    # 마지막 단계에서 남은 거리를 추가
    remaining_distance_x = end_point[0] - (start_point[0] + step_x * num_steps)
    remaining_distance_y = end_point[1] - (start_point[1] + step_y * num_steps)

    action_sequence.append((remaining_distance_x, remaining_distance_y))

    return np.array(action_sequence), len(action_sequence)

import matplotlib.pyplot as plt

def plot_trajectory(start_point, action_sequence):
    x_values = [start_point[0]]
    y_values = [start_point[1]]

    current_point = start_point

    for action in action_sequence:
        dx, dy = action
        new_x = current_point[0] + dx
        new_y = current_point[1] + dy
        x_values.append(new_x)
        y_values.append(new_y)
        current_point = (new_x, new_y)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.xlabel('X 좌표')
    plt.ylabel('Y 좌표')
    plt.title('시작점에서 끝점으로 이동한 궤적')
    plt.grid(True)
    plt.show()

# 예제 사용
start_point = np.array([1.0, 2.0])  # 시작점 좌표 (x, y)
end_point = (4.0, 16.0)    # 끝점 좌표 (x, y)
max_distance = (0.04, 0.04)  # 최대 이동 거리 (dx, dy)

action_sequence, _ = generate_action_sequence(start_point, end_point, max_distance)

plot_trajectory(start_point, action_sequence)
