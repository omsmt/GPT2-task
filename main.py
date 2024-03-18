from app.render import Render

class Application:
    def __init__(self):
        self.render = Render()  # Instantiate UIClass

    def main(self):
        self.render.main_ui()  # Call the method on the instance

if __name__ == '__main__':
    app = Application()  # Instantiate Application
    app.main()