from pynput import keyboard
import time 

# 각 키에 대한 상태를 추적하기 위한 변수
c_key_pressed = False
o_key_pressed = False

# 키가 눌렸을 때 실행할 함수
def on_press(key):
    global c_key_pressed, o_key_pressed
    if key == keyboard.KeyCode.from_char('c'):
        c_key_pressed = True
    elif key == keyboard.KeyCode.from_char('o'):
        o_key_pressed = True

# 키가 놓였을 때 실행할 함수
def on_release(key):
    global c_key_pressed, o_key_pressed
    if key == keyboard.KeyCode.from_char('c'):
        c_key_pressed = False
    elif key == keyboard.KeyCode.from_char('o'):
        o_key_pressed = False

# 키보드 리스너 시작
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

i=0



# 메인 루프
while True:
    i+=1
    if not(c_key_pressed or o_key_pressed):
        print(i)

    if c_key_pressed:
        # 'c' 키를 눌러 그리퍼를 닫기
        print("close_gripper")
    
    if o_key_pressed:
        # 'o' 키를 눌러 그리퍼를 열기
        print("open_gripper")

    # CPU 사용을 줄이기 위한 짧은 딜레이
    time.sleep(0.05)

# 리스너 종료
listener.stop()
