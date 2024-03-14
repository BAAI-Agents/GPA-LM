def take_cover(duration=2):
    """
    press "Q" to take cover
    """
    io_env.key_press('q', duration)

def focus_on_horse():
    """
    hold "right mouse button" to focus on the horse
    """
    io_env.mouse_hold('right mouse button')

def view_stored_weapons():
    """
    press "tab" to view your stored weapons
    """
    io_env.key_hold('tab')