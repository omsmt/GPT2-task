from app.gpt2_model import GPT2_model
import threading
import imgui

class UIClass():
    def __init__(self):
        self.gpt2 = GPT2_model()
        self.text = ""
        self.output_text = None
        self.is_generating = False
        self.is_loading = False
        self.loaded_model = None
        self.model_sizes = ['gpt2', 'gpt2-medium', 'gpt2-large']
        self.current_model_size_index = 1  # Use index directly
        self.current_model_size = self.model_sizes[self.current_model_size_index]
        self.quantise = True

    def ui_loop(self):

        # Begin ImGui frame
        imgui.begin("Let's Get Generating!", flags=imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)

        imgui.dummy(5,5)

        imgui.text_wrapped("""Welcome to this humble UI that is housing the unassuming star of the show, 
                           \nGPT2! Well, GPT2-Medium to be specific.
                           \nWhat does this mean? Great question! You can enter a prompt, and this autogressive decoding model will create a continuation of it. 
                           \nFirst however, let's select and then load the model by pressing the button below. Patience will be a virtue for this step.""")

        imgui.dummy(5,5)

        # Set the combo box width
        desired_width = 125  # Set this to your desired width
        imgui.push_item_width(desired_width)

        clicked, self.current_model_size_index = imgui.combo(
            "##Model Size", self.current_model_size_index, self.model_sizes
        )
        if clicked:
            self.current_model_size = self.model_sizes[self.current_model_size_index]

        imgui.pop_item_width()  # Revert to the previous item width

        _, self.quantise = imgui.checkbox("Quantise model?", self.quantise)

        if self.is_loading:
            imgui.button("Loading Model and Weights...")
        else:
            if imgui.button("Load Model and Weights"):
                self.is_loading = True
                thread = threading.Thread(target=self.load_model_async, args=(self.current_model_size, self.quantise), daemon=True)
                thread.start()

        if self.gpt2.model is not None and not self.is_loading:
            imgui.same_line()
            imgui.text(f"{self.loaded_model} model successfully loaded!")

            imgui.dummy(5,5)
            imgui.separator()
            imgui.dummy(5,5)

            imgui.text("Enter prompt below:")

            # Input text section with an Enter key trigger
            enter_pressed = False
            changed, self.text = imgui.input_text("##Input", self.text, 256, imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed:
                # Reset output text when input changes
                self.output_text = None
                self.inf_time = None
                enter_pressed = True

            imgui.same_line()

            if enter_pressed or imgui.button("Generate" if not self.is_generating else "Generating..."):
                if not self.is_generating:
                    self.is_generating = True
                    threading.Thread(target=self.generate_text_async).start()

            imgui.dummy(5,5)

            # Text box for output
            imgui.text("Output will be shown here:")

            if self.output_text is not None:
                imgui.text_wrapped(self.output_text)

                imgui.dummy(5,5)

                imgui.text_wrapped(f'Model inference time: {self.inf_time:.2f}s')
            else:
                imgui.text_wrapped("")  # or some placeholder text like "Output will appear here after generating."

        imgui.end()

    def load_model_async(self, model_size, quantise):
        self.gpt2.load_model_and_weights(model_size, quantise)
        self.is_loading = False
        self.loaded_model = model_size

    def generate_text_async(self):
        self.output_text, self.inf_time = self.gpt2.inference(self.text)
        self.is_generating = False




