"""
TCP Camera Application
-----------------------
This script interfaces with Toshiba cameras using the pytelicam SDK.
It captures live video streams from multiple connected cameras and displays
them using OpenCV. The application handles camera initialization, stream configuration,
image acquisition, and cleanup.

Commands:
- "start" starts recording
- "stop" stops recording
- "exit" exits the application
- Displays resized 320x240 preview windows
"""
import sys  # Import system-specific parameters and functions
import numpy as np  # Import NumPy for numerical operations
import cv2  # Import OpenCV for image processing
import pytelicam  # Import the pytelicam SDK for camera control
import threading  # Import threading for concurrent execution
from datetime import datetime  # Import datetime for timestamping recordings
from math import floor
import traceback
import os  # Import os for file operations
import time

class Recorder:
    def __init__(self):
        self.recording = False  # Flag to indicate if recording is in progress
        self.writers = []  # List to hold video writer objects
        self.stop_event = threading.Event()  # Event to signal stopping of threads
        self.thread = None  # Thread for capturing frames
        self.displaying = False  # Flag to indicate if camera display is active
        self.display_windows = []  # List to hold display window names
        self.filenames = []
        self.width = 1224
        self.height = 1024
        self.fps = 25
        self.dB = 19.5
        self.xOffset = int(float(self.width) / 2.0)
        self.yOffset = int(float(self.height) / 2.0)

        # Initialize the camera system using the U3V interface
        self.cam_system = pytelicam.get_camera_system(int(pytelicam.CameraType.U3v))
        self.cam_devices = []  # List to hold camera device objects
        self.receive_signals = []  # List to hold signal objects for each camera
        self.cam_num = self.cam_system.get_num_of_cameras()  # Get the number of connected cameras
        if self.cam_num == 0:  # Check if no cameras are found
            print("No cameras found. Exiting application.")  # Print an error message
            sys.exit()  # Exit the application gracefully
        else:
            print(f"Detected {self.cam_num} camera(s)")

        # Create device objects and signal objects for each camera
        for i in range(self.cam_num):
            self.cam_devices.append(self.cam_system.create_device_object(i))
            self.receive_signals.append(self.cam_system.create_signal())
        
        # Open each camera device and configure it
        for i,device in enumerate(self.cam_devices):
            device.open()  # Open the camera device

            res = device.cam_control.set_trigger_mode(False)  # Disable hardware trigger mode for continuous acquisition
            if res != pytelicam.CamApiStatus.Success:
                raise Exception("Can't set TriggerMode.")  # Raise an exception if unable to set trigger mode

            res = self.cam_devices[i].cam_control.set_width(self.width)  # set the width of the camera feed
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set width. {res}")

            res = self.cam_devices[i].cam_control.set_height(self.height)  # set the height of the camera feed
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set height. {res}")

            res = self.cam_devices[i].cam_control.set_acquisition_frame_rate_control(pytelicam.pytelicam.CameraAcqFrameRateCtrl.Manual)
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set cam ctrl. {res}")

            res = self.cam_devices[i].cam_control.set_acquisition_frame_rate(self.fps)  # set the frame rate of the camera
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set aquisition fps. Camera {i} | {res}")

            #debug stuff
            res, mode = self.cam_devices[i].cam_control.get_acquisition_frame_rate_control()
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set cam ctrl. {res}")
            print(mode) #debugging

            res,fps = self.cam_devices[i].cam_control.get_acquisition_frame_rate()  # set the frame rate of the camera
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set aquisition fps. Camera {i} | {res}")
            print(fps) #debugging
            #end debug

            res = self.cam_devices[i].cam_control.set_offset_x(self.xOffset)
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set xoffset setting. Camera {i} | {res}")

            res = self.cam_devices[i].cam_control.set_offset_y(self.yOffset)
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set yoffset setting. Camera {i} | {res}")

            res = self.cam_devices[i].cam_control.set_gain(self.dB)
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set gain auto setting. Camera {i} | {res}")

            res = self.cam_devices[i].cam_control.set_balance_white_auto(
                pytelicam.CameraBalanceWhiteAuto.Once)
            if res != pytelicam.CamApiStatus.Success:
                raise Exception(f"Can't set white balance auto setting. Camera {i} | {res}")

            device.cam_stream.open(self.receive_signals[i])  # Open the camera stream
            device.cam_stream.start()  # Start the camera stream

        self.start_display()  # Start displaying the camera feeds

    def start_display(self):
        # Start displaying the camera feeds in separate windows
        if self.displaying:
            print("Already displaying cameras!")  # Inform the user if already displaying
            return

        self.displaying = True  # Set the displaying flag to true
        self.display_windows = [f"Camera {i}" for i in range(self.cam_num)]  # Create window names for each camera
            
        self.display_thread = threading.Thread(target=self._update_displays)  # Create a thread to update displays
        self.display_thread.start()  # Start the display thread
        print("Camera display started - type 'stop' to exit")  # Inform the user that display has started

    def _update_displays(self):
        # Continuously update the display windows with frames from the cameras
        while self.displaying:
            for i, window in enumerate(self.display_windows):
                res = self.cam_system.wait_for_signal(self.receive_signals[i])  # Wait for a signal from the camera
                if res == pytelicam.CamApiStatus.Success:
                    current_index = self.cam_devices[i].cam_stream.get_current_buffer_index()
                    with self.cam_devices[i].cam_stream.get_buffered_image(current_index) as image_data:
                        if image_data.status == pytelicam.CamApiStatus.Success:
                            frame = None
                            if image_data.pixel_format == pytelicam.CameraPixelFormat.Mono8:
                                frame = image_data.get_ndarray(pytelicam.OutputImageType.Raw)
                            else:
                                frame = image_data.get_ndarray(pytelicam.OutputImageType.Bgr24) # Get the current frame as a NumPy array
                            frame = cv2.resize(frame, dsize=(320, 240))
                            cv2.imshow(window, frame)  # Display the frame in the corresponding window
                        else:
                            print("")
                            #print(f"Grab error! status = {image_data.status} camera: {i}")
                else:
                    print(f"Signal error ! status = {res} camera: {i}")
            cv2.waitKey(1)

    def start_recording(self, w =2448, h =2048):
        # Start recording video from the cameras
        if self.recording:
            print("Already recording!")  # Inform the user if already recording
            return

        try:
            self.writers = []  # Reset the writers list
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get the current timestamp for file naming
            for i in range(self.cam_num):
                filename = f"output/recording_cam{i}_{timestamp}.mp4"  # Create a filename for the recording
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the video writer
                self.writers.append(cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height)))  # Create a video writer for each camera
                self.filenames.append(filename)

            
            self.recording = True  # Set the recording flag to true
            self.stop_event.clear()  # Clear the stop event
            self.thread = threading.Thread(target=self._capture_frames)  # Create a thread to capture frames
            self.thread.start()  # Start the frame capturing thread
            print("Recording started...")  # Inform the user that recording has started
            
        except Exception as e:
            print(f"Failed to start recording: {str(e)}")  # Handle any exceptions that occur during recording initialization
            print(traceback.format_exc())
            self.writers = []  # Reset the writers list
            
    def _capture_frames(self):
        # Capture frames from the cameras while recording
        while self.recording and not self.stop_event.is_set():
            for i, window in enumerate(self.display_windows):
                res = self.cam_system.wait_for_signal(self.receive_signals[i])  # Wait for a signal from the camera
                if res == pytelicam.CamApiStatus.Success:
                    with self.cam_devices[i].cam_stream.get_current_buffered_image() as image_data:
                        if image_data.status == pytelicam.CamApiStatus.Success:
                            if image_data.pixel_format == pytelicam.CameraPixelFormat.Mono8:
                                frame = image_data.get_ndarray(pytelicam.OutputImageType.Raw)
                                self.writers[i].write(frame)  # Write the frame to the corresponding video file
                            else:
                                frame = image_data.get_ndarray(pytelicam.OutputImageType.Bgr24)  # Get the current frame as a NumPy array
                                self.writers[i].write(frame)  # Write the frame to the corresponding video file

    def stop_display(self):
        # Stop displaying the camera feeds
        if self.displaying:
            self.displaying = False  # Set the displaying flag to false
            #self.stop_event.set()  # Signal to stop the display thread
            if self.display_thread.is_alive():
                self.display_thread.join()  # Wait for the display thread to finish
            for window in self.display_windows:
                cv2.destroyWindow(window)  # Close all display windows
            self.display_windows = []  # Clear the list of display windows
            print("Camera display stopped")  # Inform the user that display has stopped

    def stop_recording(self, save=True):
        self.recording = False  # Set the recording flag to false

        if self.thread.is_alive():
            self.thread.join()  # Wait for the thread to finish

        self.stop_event.set()  # Signal to stop the thread
        
        for writer in self.writers:
            writer.release()  # Release the video writer resources
        
        if not save:
            for file in self.filenames:
                if os.path.exists(file):
                    os.remove(file)  # Delete the temporary video file if not saving
        
        for file in self.filenames:
            print("Recording stopped" + (f" and saved in {file}" if save else " (discarded)"))  # Inform the user of the recording status
        
        self.writers = []  # Reset the writers list

    def cleanup(self):
        self.stop_display()  # Stop displaying the camera feeds
        # Cleanup resources and terminate the camera system
        for i in range(self.cam_num):
            if self.cam_devices[i] is not None:
                if self.cam_devices[i].cam_stream.is_open:
                    self.cam_devices[i].cam_stream.stop()
                    self.cam_devices[i].cam_stream.close()

                if self.cam_devices[i].is_open:
                    self.cam_devices[i].close()

            if self.receive_signals[i] is not None:
                self.cam_system.close_signal(self.receive_signals[i])

        self.cam_system.terminate()

        print("Finished.")

def handle_save(r):
    while True:
        save = input("Save recording? (yes/no): ").lower().strip()  # Ask the user if they want to save the recording
        if save in ["yes", "no"]:
            r.stop_recording(save == "yes")  # Stop recording and save if the user confirms
            break
        else:
            print("Please enter 'yes' or 'no'.")  # Prompt for valid input

"""
#implement later
def handle_start_recording(r):
    w, h = 2448, 2048
    while True:
        default = input("Would you like the default width and height for the recording? (yes/no): ").lower().strip()
        if default in ["yes", "no"]:
            r.start_recording()
            break
        else:
            print("Please enter 'yes' or 'no'.")
"""



def del_input(cmd):
    if cmd == "start":
        recorder.start_recording()  # Start recording when the user enters "start"
    elif cmd == "stop":
        if recorder.recording:
            handle_save(recorder)
        else:
            print("Not recording")  # Inform the user if not currently recording
    elif cmd == "exit":
        if recorder.recording:
            handle_save(recorder)
        return False
    elif cmd == "debug_exit":
        return False
    else:
        print("Invalid command")  # Inform the user of invalid input

    return True

if __name__ == "__main__":
    try:
        recorder = Recorder()  # Create an instance of the Recorder class
        print("Camera Control REPL...\nCommands: start, stop, exit")  # Display available commands
        while True:
            cmd = input("> ").lower().strip()  # Get user input
            if del_input(cmd) == False: break

    except pytelicam.PytelicamError as teli_exception:
        print("An error occurred!")
        print(f"  message : {teli_exception.message}")
        print(f"  status  : {teli_exception.status}")
        print(traceback.format_exc())
    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Handle any exceptions that occur during execution
        print(traceback.format_exc())
    finally:
        if 'recorder' in locals():
            recorder.cleanup()  # Ensure cleanup is called on error

