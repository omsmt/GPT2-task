from app.ui import UIClass
import os
import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

class Render:
    def __init__(self):
        self.ui = UIClass()
        self.font_path = 'fonts/Geneva.ttf'

    def window_positioning(self, window):
        # Get the current size of the GLFW window
        window_width, window_height = glfw.get_window_size(window)

        # Calculate the desired ImGui window size
        imgui_window_width = window_width - 50
        imgui_window_height = window_height - 50

        # Start new frame
        imgui.new_frame()

        # Set the size and position of the ImGui window to make it smaller
        # than the GLFW window by 50 units in each dimension
        imgui.set_next_window_size(imgui_window_width, imgui_window_height)
        imgui.set_next_window_position(25, 25)  # Offset by 25 to center it


    def glfw_initialising(self):
        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            return

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    def main_ui(self):

        self.glfw_initialising()

        # Create a GLFW window
        window = glfw.create_window(600, 500, "Mia's Technical Task", None, None)
        glfw.make_context_current(window)

        # Initialize ImGui and the GLFW renderer
        imgui.create_context()
        impl = GlfwRenderer(window)

        # Set custom font and size (ensure the font file path is correct)
        io = imgui.get_io()
        if os.path.exists(self.font_path):  # Check if the font file exists
            font = io.fonts.add_font_from_file_ttf(self.font_path, 15)
            impl.refresh_font_texture()
        else:
            print(f"Font file not found: {self.font_path}")

        # Main loop
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            self.window_positioning(window)

            with imgui.font(font):
                self.ui.ui_loop()

            # Rendering
            imgui.render()
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)

        impl.shutdown()
        glfw.terminate()