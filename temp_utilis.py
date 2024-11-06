from termcolor import colored

# deprecated
def logger(message, color="green"):
    print(colored(f"[DEBUG] {message}", color))

def bug(message, color="yellow"):
    if isinstance(message, tuple):
        # Unpack the tuple
        msg, var = message
        print(colored(f"[DEBUG] {msg} : {var}", color))
    elif isinstance(message, str):
        # Just print the string message
        print(colored(f"[DEBUG] {message}", color))
    else:
        # Handle invalid input
        print(colored("[DEBUG] Invalid message format", "red"))

