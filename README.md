# Moovier: Movie Recommender System

## Introduction
Moovier is a comprehensive movie recommendation system designed to offer personalized movie suggestions to users. This project is split into two main components: a robust backend API with a sophisticated machine learning model, and a user-friendly frontend interface.

### Features

- Personalized movie recommendations based on user preferences and interaction.
- Advanced machine learning algorithms for accurate suggestions.
- User-friendly API for seamless integration and interaction.
- Intuitive frontend interface for an engaging user experience.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
TODO: List all the tools or libraries that are necessary to run your project:
- Example: Python 3.8+
- Example: Node.js and React (for the frontend)


### Installation
Step by step series of examples that tell you how to get a an environment running.

1. Clone the backend repo:
   ```bash
   git clone https://github.com/moovier/backend.git
   ```
2. Clone the frontend repo:
   ```bash
   git clone https://github.com/moovier/frontend.git
   ```
3. Navigate to the project backend directory:
    ```bash
    cd moovier/backend
    ```
4. Create environment and install the required packages:
    ```bash
    conda create --name moovier python=3.8+
    conda activate moovier
    pip install -r src/requirements.txt
    ```
5. Navigate to the frontend directory:
    ```bash
    cd ../frontend
    ```
6.  
    ```bash
    npm install
    ```

### Usage

#### Starting the Backend Server
```bash
python path/to/magic.py
```

#### (Optional?)Run kedro pipeline
You can run your Kedro project with:

```bash
cd backend
kedro run --env=base
```

#### Starting the Fronend Application
```bash
cd frontend
npm start
```
Verify the application is running by navigating to your server address in your preferred browser\
http://localhost:3000/

## Authors
- **Leonardo Wajda** - *Initial work* - [YourUsername](https://github.com/leowajda)
- **Azimjon Pulatov** - *Initial work* - [YourUsername](https://github.com/azimjohn)
- **Lizaveta Babior** - *Initial work* - [YourUsername](https://github.com/Babior)

## License
This project is licensed under the XYZ License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
- Hat tip to anyone whose code was used (fronend guy)
- Inspiration (netflix)
- etc