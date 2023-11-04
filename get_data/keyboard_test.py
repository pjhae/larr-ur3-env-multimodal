from pynput import keyboard
import time

# This flag will be used to determine if a key is pressed
key_pressed = False

# The event listener will set this flag to True when a key is pressed
def on_press(key):
    global key_pressed
    key_pressed = True

# Start the listener in the background
listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    print("1")

    if key_pressed:
        print("2")
        key_pressed = False

    # Add a small delay to reduce CPU usage
    time.sleep(0.5)

# Stop the listener when done
listener.stop()



# import keyboard
# import time

# for i in range(100):  # 예를 들어 100번 반복
#     if keyboard.is_pressed('q'):  # 'q' 키가 눌렸는지 확인
#         print(2)
#     else:
#         print(1)
#     time.sleep(0.1)  # CPU 사용률을 낮추기 위해 잠시 대기



