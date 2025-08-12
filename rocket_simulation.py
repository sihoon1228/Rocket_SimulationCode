import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # GUI 환경에서 실행할 경우
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# === 파라미터 ===
g = 9.81               # 중력가속도(m/s^2)
m0 = 0.5               # 초기 질량(kg)
dm_dt = -0.02          # 질량 유량(kg/s) → 연료 소모 속도
A = 0.02               # 단면적(m^2)
C_D = 0.75             # 항력 계수
rho0 = 1.225           # 해수면 공기 밀도(kg/m^3)
H = 8500               # 대기 밀도 감소 고도(m)
burn_time = 3      # 연소 시간(s)
thrust_magnitude = 20  # 추진력(N)

theta_deg = 20         # 고각
phi_deg = 30           # 방위각
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

wind_velocity = np.array([0.0, 2.0, 0.0])  # [동, 북, 상] 바람 속도(m/s)

# === 시간 설정 ===
t_span = (0, 30)
t_eval = np.linspace(*t_span, 500)

# === 추력 벡터 ===
def thrust_vector(t):
    if t > burn_time:
        return np.zeros(3)
    return thrust_magnitude * np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        np.sin(theta)
    ])

# === 운동 방정식 ===
def rocket_derivatives(t, y):
    r = y[:3]      # 위치
    v = y[3:]      # 속도
    h = r[2]       # 고도
    m = max(m0 + dm_dt * t, 0.1)  # 시간에 따른 질량 (최소값 제한)
    rho = rho0 * np.exp(-h / H)   # 고도에 따른 공기 밀도
    v_rel = v - wind_velocity     # 상대 속도

    F_gravity = np.array([0, 0, -m * g])
    F_drag = -0.5 * rho * C_D * A * np.linalg.norm(v_rel) * v_rel
    F_thrust = thrust_vector(t)
    pseudo_force = - (dm_dt / m) * v  # 질량 손실에 따른 추가 항

    a = (F_thrust + F_gravity + F_drag) / m + pseudo_force
    return np.concatenate((v, a))

# === 초기 조건 ===
y0 = np.array([0, 0, 0, 0, 0, 0])
sol = solve_ivp(rocket_derivatives, t_span, y0, t_eval=t_eval)
x, y, z = sol.y[0], sol.y[1], sol.y[2]

# === 비행 정보 ===
landing_index = np.where(z >= 0)[0][-1]
flight_time = sol.t[landing_index]
max_altitude = np.max(z)
max_altitude_index = np.argmax(z)
horizontal_distance = np.linalg.norm([x[landing_index], y[landing_index]])

# === 애니메이션 설정 ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.min(x) * 1.1, np.max(x) * 1.1)
ax.set_ylim(np.min(y) * 1.1, np.max(y) * 1.1)
ax.set_zlim(0, np.max(z) * 1.1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Rocket Trajectory (θ={theta_deg}°, φ={phi_deg}°)')

trajectory_line, = ax.plot([], [], [], 'b-', lw=2)
rocket_point, = ax.plot([], [], [], 'ro')
text_box = ax.text2D(0.05, 0.80, "", transform=ax.transAxes)

# 최고고도 표시
peak_marker = ax.scatter([x[max_altitude_index]], [y[max_altitude_index]], [z[max_altitude_index]], c='g', s=50)
peak_label = ax.text(
    x[max_altitude_index], y[max_altitude_index], z[max_altitude_index],
    f'{z[max_altitude_index]:.2f} m',
    fontsize=10, color='green'
)

animation_running = True

def update(frame):
    global animation_running
    if not animation_running:
        return trajectory_line, rocket_point, text_box

    trajectory_line.set_data(x[:frame], y[:frame])
    trajectory_line.set_3d_properties(z[:frame])
    rocket_point.set_data([x[frame]], [y[frame]])
    rocket_point.set_3d_properties([z[frame]])

    if frame >= landing_index:
        text_box.set_text(
            f"Max Altitude: {max_altitude:.2f} m\n"
            f"Flight Time: {flight_time:.2f} s\n"
            f"Horizontal Distance: {horizontal_distance:.2f} m"
        )
        ani.event_source.stop()
        animation_running = False

    return trajectory_line, rocket_point, text_box, peak_marker, peak_label

ani = FuncAnimation(
    fig, update,
    frames=range(1, len(t_eval)),
    interval=20, blit=False
)

plt.tight_layout()
plt.show()