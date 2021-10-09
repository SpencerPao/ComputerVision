import win32api as wapi
import win32con as con
# keyList = [' ', con.VK_UP, con.VK_DOWN]
keyList = [32, 'Q', con.VK_UP, con.VK_DOWN]  # 32 is space.


def keys():
    '''Retrieves the associated key with snapshot.'''
    keys_array = []
    for key in keyList:
        if isinstance(key, int):
            if wapi.GetAsyncKeyState(key):
                keys_array.append(key)
        elif isinstance(key, str):
            if wapi.GetAsyncKeyState(ord(key)):
                keys_array.append(key)
    if 32 in keys_array:  # Key for Space: ''
        return 32
    elif 'Q' in keys_array:
        return 'Q'
    elif con.VK_UP in keys_array:
        return con.VK_UP
    elif con.VK_DOWN in keys_array:
        return con.VK_DOWN
