# Furkan'in yazdigi komut kodu

import os
import ctypes
import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from ctypes.wintypes import HWND, UINT
import subprocess

def ses_kapat():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(1, None)

def ses_ac():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(0, None)

def sesi_yukselt():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume.SetMasterVolumeLevelScalar(min(1.0, current_volume + 0.1), None)

def sesi_azalt():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume.SetMasterVolumeLevelScalar(max(0.0, current_volume - 0.1), None)

def parlaklik_artir():
    current_brightness = int(subprocess.check_output(
        "powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"))
    new_brightness = min(current_brightness + 10, 100)
    subprocess.run(["powershell",
                    f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{new_brightness})"])

def parlaklik_azalt():
    current_brightness = int(subprocess.check_output(
        "powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"))
    new_brightness = max(current_brightness - 10, 0)
    subprocess.run(["powershell",
                    f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{new_brightness})"])

def bilgisayari_kilitle():
    pyautogui.hotkey('winleft', 'd')

def muzik_video_oynat_durdur():
    pyautogui.press("playpause")

def slayt_ileri():
    pyautogui.press("right")

def slayt_geri():
    pyautogui.press("left")

def main():
    while True:
        print("1. Ses kapatma")
        print("2. Ses Açma")
        print("3. Sesi Yükselt, her tıklamada %10")
        print("4. Sesi Azalt, her tıklamada %10")
        print("5. Ekran Parlaklığını Artır, her tıklamada %10")
        print("6. Ekran Parlaklığını Azalt, her tıklamada %10")
        print("7. Ana Ekrana Dön")
        print("8. Müzik/Video Oynat-Durdur")
        print("9. Slayt İleri")
        print("10. Slayt Geri")
        print("0. Çıkış")

        choice = input("Bir komut seçin (1-10): ")

        if choice == "1":
            ses_kapat()
        elif choice == "2":
            ses_ac()
        elif choice == "3":
            sesi_yukselt()
        elif choice == "4":
            sesi_azalt()
        elif choice == "5":
            parlaklik_artir()
        elif choice == "6":
            parlaklik_azalt()
        elif choice == "7":
            bilgisayari_kilitle()
        elif choice == "8":
            muzik_video_oynat_durdur()
        elif choice == "9":
            slayt_ileri()
        elif choice == "10":
            slayt_geri()
        elif choice == "0":
            break
        else:
            print("Geçersiz seçim, lütfen tekrar deneyin.")


if __name__ == "__main__":
    main()