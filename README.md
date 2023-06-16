
# README 

## AI Image and Text Understanding Application

This AI application is a combination of image and text understanding tasks. It performs various operations like image classification, segmentation, object detection, image captioning, emotion detection, text summarization, and text generation.

### Structure of the Application

The main Python file is `app.py` which uses Flask to serve an API with different endpoints for image and text processing tasks. It takes the input image or text from the user and processes it using various AI models to output the results. These results are then returned as a JSON response to the client. 

The client is a HTML page that has a user interface for users to interact with the AI application. The HTML page contains a form for uploading images or entering text, and scripts for calling the API endpoints, processing the returned results, and displaying them on the webpage.

## Instructions for Running the Application

### Prerequisites

- Python 3.7 or higher
- Docker installed on your machine

### Steps

1. Clone this repository to your local machine.

2. Build the Docker image by running the following command in your terminal from the directory where the Dockerfile is located:

```sh
docker build -t ai_app .
```



3. Run the Docker container:

```sh
docker run -p 5020:5020 ai_app
```

4. Open your web browser and visit `http://localhost:5020/` to access the application.

### Using the Application

The application has an HTML interface where you can upload images or enter text.

- For image processing, upload an image and click the "Process Image" button. The application will process the image and display the results which include the top five classification predictions, a segmented version of the image, detected objects in the image, and a caption for the image.

- For text processing, enter some text into the text area and click the "Process Text" button. The application will process the text and display the results which include the detected emotion, a summary of the text, and some generated text based on the input.

Enjoy using the application!