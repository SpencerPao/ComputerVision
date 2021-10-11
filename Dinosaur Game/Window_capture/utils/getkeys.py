"""Get Keys from keyboard."""
import win32api as wapi
import win32con as con
keyList = [con.VK_SPACE, 0x51, con.VK_UP, con.VK_DOWN]


def keys():
    """Retrieves the associated key with snapshot."""
    keys_array = []
    for key in keyList:
        if isinstance(key, int):
            if wapi.GetAsyncKeyState(key):
                keys_array.append(key)
    if con.VK_SPACE in keys_array:
        return con.VK_SPACE
    elif 0x51 in keys_array:  # 'Q' for quit.
        return 0x51
    elif con.VK_UP in keys_array:
        return con.VK_UP
    elif con.VK_DOWN in keys_array:
        return con.VK_DOWN
