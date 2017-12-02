import win32api as w_api


KL=["\b"]
for c in "123456789":
    KL.append(c)
    
def k_check():
    key=[]
    for k in KL:
        if w_api.GetAsyncKeyState(ord(k)):
            key.append(k)
    return key        
