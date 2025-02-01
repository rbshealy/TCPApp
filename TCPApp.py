"""
TCP Camera Application
-----------------------
This script interfaces with Toshiba cameras using the pytelicam SDK.
It captures live video streams from multiple connected cameras and displays
them using OpenCV. The application handles camera initialization, stream configuration,
image acquisition, and cleanup.

Controls:
- ESC or 'q' key exits the application
- Displays resized 320x240 preview windows
"""
import sys  # Import system-specific parameters and functions
import numpy as np  # Import NumPy for numerical operations
import cv2  # Import OpenCV for image processing
import pytelicam  # Import the pytelicam SDK for camera control
import threading  # Import threading for concurrent execution
from datetime import datetime  # Import datetime for timestamping recordings

class Recorder:
    def __init__(self):
        # Initialize the camera system using the U3V interface
        self.cam_system = pytelicam.get_camera_system(int(pytelicam.CameraType.U3v))
        self.cam_devices = []  # List to hold camera device objects
        self.receive_signals = []  # List to hold signal objects for each camera
        self.cam_num = self.cam_system.get_num_of_cameras()  # Get the number of connected cameras
        if self.cam_num == 0:  # Check if no cameras are found
            print("No cameras found. Exiting application.")  # Print an error message
            sys.exit()  # Exit the application gracefully
        
        # Create device objects and signal objects for each camera
        for i in range(self.cam_num):
            self.cam_devices.append(self.cam_system.create_device_object(i))
            self.receive_signals.append(self.cam_system.create_signal())
        
        # Open each camera device and configure it
        for device in self.cam_devices:
            device.open()  # Open the camera device
            res = device.cam_control.set_trigger_mode(False)  # Disable hardware trigger mode for continuous acquisition
            if res != pytelicam.CamApiStatus.Success:
                raise Exception("Can't set TriggerMode.")  # Raise an exception if unable to set trigger mode
            device.cam_stream.open(self.receive_signals[i])  # Open the camera stream
            device.cam_stream.start()  # Start the camera stream

        self.recording = False  # Flag to indicate if recording is in progress
        self.writers = []  # List to hold video writer objects
        self.stop_event = threading.Event()  # Event to signal stopping of threads
        self.thread = None  # Thread for capturing frames
        self.displaying = False  # Flag to indicate if camera display is active
        self.display_windows = []  # List to hold display window names

    def start_display(self):
        # Start displaying the camera feeds in separate windows
        if self.displaying:
            print("Already displaying cameras!")  # Inform the user if already displaying
            return

        self.displaying = True  # Set the displaying flag to true
        self.display_windows = [f"Camera {i}" for i in range(self.cam_num)]  # Create window names for each camera
        for window in self.display_windows:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # Create a named window for each camera
            cv2.resizeWindow(window, 320, 240)  # Resize the window to 320x240 pixels
            
        self.display_thread = threading.Thread(target=self._update_displays)  # Create a thread to update displays
        self.display_thread.start()  # Start the display thread
        print("Camera display started - type 'stop' to exit")  # Inform the user that display has started

    def _update_displays(self):
        # Continuously update the display windows with frames from the cameras
        while self.displaying and not self.stop_event.is_set():
            for i, window in enumerate(self.display_windows):
                res = self.cam_system.wait_for_signal(self.receive_signals[i], 100)  # Wait for a signal from the camera
                if res == pytelicam.CamApiStatus.Success:
                    with self.cam_devices[i].cam_stream.get_current_buffered_image() as image_data:
                        if image_data.status == pytelicam.CamApiStatus.Success:
                            frame = image_data.get_ndarray(pytelicam.OutputImageType.Bgr24)  # Get the current frame as a NumPy array
                            cv2.imshow(window, frame)  # Display the frame in the corresponding window
                            cv2.waitKey(1)  # Wait for a short period to allow the window to update

    def start_recording(self):
        # Start recording video from the cameras
        if self.recording:
            print("Already recording!")  # Inform the user if already recording
            return
        
        try:
            self.writers = []  # Reset the writers list
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get the current timestamp for file naming
            for i in range(self.cam_num):
                width = self.cam_devices[i].get_feature_value("Width")  # Get the width of the camera feed
                height = self.cam_devices[i].get_feature_value("Height")  # Get the height of the camera feed
                fps = self.cam_devices[i].get_feature_value("AcquisitionFrameRate")  # Get the frame rate of the camera
                filename = f"recording_cam{i}_{timestamp}.mp4"  # Create a filename for the recording
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the video writer
                self.writers.append(cv2.VideoWriter(filename, fourcc, fps, (width, height)))  # Create a video writer for each camera
            
            self.recording = True  # Set the recording flag to true
            self.stop_event.clear()  # Clear the stop event
            self.thread = threading.Thread(target=self._capture_frames)  # Create a thread to capture frames
            self.thread.start()  # Start the frame capturing thread
            self.start_display()  # Start displaying the camera feeds
            print("Recording started")  # Inform the user that recording has started
            
        except Exception as e:
            print(f"Failed to start recording: {str(e)}")  # Handle any exceptions that occur during recording initialization
            self.writers = []  # Reset the writers list
            
    def _capture_frames(self):
        # Capture frames from the cameras while recording
        while self.recording and not self.stop_event.is_set():
            for i in range(self.cam_num):
                res = self.cam_system.wait_for_signal(self.receive_signals[i], 100)  # Wait for a signal from the camera
                if res == pytelicam.CamApiStatus.Success:
                    with self.cam_devices[i].cam_stream.get_current_buffered_image() as image_data:
                        if image_data.status == pytelicam.CamApiStatus.Success:
                            frame = image_data.get_ndarray(pytelicam.OutputImageType.Bgr24)  # Get the current frame as a NumPy array
                            self.writers[i].write(frame)  # Write the frame to the corresponding video file

    def stop_display(self):
        # Stop displaying the camera feeds
        if self.displaying:
            self.displaying = False  # Set the displaying flag to false
            self.stop_event.set()  # Signal to stop the display thread
            if self.display_thread.is_alive():
                self.display_thread.join()  # Wait for the display thread to finish
            for window in self.display_windows:
                cv2.destroyWindow(window)  # Close all display windows
            self.display_windows = []  # Clear the list of display windows
            print("Camera display stopped")  # Inform the user that display has stopped

    def stop_recording(self, save=True):
        self.recording = False  # Set the recording flag to false
        self.stop_event.set()  # Signal to stop the thread
        if self.thread.is_alive():
            self.thread.join()  # Wait for the thread to finish
        
        self.stop_display()  # Stop displaying the camera feeds
        
        for writer in self.writers:
            writer.release()  # Release the video writer resources
        
        if not save:
            import os  # Import os for file operations
            for writer in self.writers:
                if os.path.exists(writer.filename):
                    os.remove(writer.filename)  # Delete the temporary video file if not saving
        
        self.writers = []  # Reset the writers list
        print("Recording stopped" + (" and saved" if save else " (discarded)"))  # Inform the user of the recording status

    def cleanup(self):
        # Cleanup resources and terminate the camera system
        for device in self.cam_devices:
            if device.is_open:
                device.cam_stream.stop()  # Stop the camera stream
                device.cam_stream.close()  # Close the camera stream
                device.close()  # Close the camera device
        self.cam_system.terminate()  # Terminate the camera system
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print("Finished.")  # Inform the user that cleanup is complete

try:
    if __name__ == "__main__":
        recorder = Recorder()  # Create an instance of the Recorder class
        print("Camera Control REPL\nCommands: start, stop, exit")  # Display available commands
        while True:
            cmd = input("> ").lower().strip()  # Get user input
            
            if cmd == "start":
                recorder.start_recording()  # Start recording when the user enters "start"
                recorder.start_display()  # Start displaying the camera feeds
            elif cmd == "stop":
                if recorder.recording:
                    while True:
                        save = input("Save recording? (yes/no): ").lower().strip()  # Ask the user if they want to save the recording
                        if save in ["yes", "no"]:
                            recorder.stop_recording(save == "yes")  # Stop recording and save if the user confirms
                            break
                        else:
                            print("Please enter 'yes' or 'no'.")  # Prompt for valid input
                else:
                    print("Not recording")  # Inform the user if not currently recording
            elif cmd == "exit":
                if recorder.recording:
                    recorder.stop_recording(False)  # Stop recording without saving if exiting
                recorder.cleanup()  # Cleanup resources before exiting
                break  # Exit the loop
            else:
                print("Invalid command")  # Inform the user of invalid input
except Exception as e:
    print(f"An error occurred: {str(e)}")  # Handle any exceptions that occur during execution
    recorder.cleanup()  # Ensure cleanup is called on error
