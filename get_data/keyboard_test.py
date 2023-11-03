import keyboard

## You should run this code with "sudo python3 ___.py"

def main():
    print("always this message when there is no input")

    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'enter':
                print("user pushed this key, :: REACT!!")


if __name__ == "__main__":
    main()

