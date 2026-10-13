import numpy as np

class DroneNeuralEngine:
    def __init__(self):

        # PART 1: Tensors, Matrices & Slicing
        self.sensor_matrix = np.array([
            [15.0, 3500.0, 120.0],
            [25.0, 3100.0, 80.0],
            [10.0, 3600.0, 150.0],
            [30.0, 2900.0, 50.0]
        ])
        
        self.labels = np.array([[1], [0], [1], [0]])
        
        # Slicing: Extracting ONLY the Altitude data 
        self.altitudes = self.sensor_matrix[:, 2]

        # Initialize AI Weights 
        np.random.seed(42)
        self.weights = np.random.rand(3, 1)

    def process_math(self):
        print("--- 1. MATRIX SLICING ---")
        print(f"Extracted Altitudes Vector: {self.altitudes}")

        # AI models crash if numbers are too big. We must scale them.
        mean = np.mean(self.sensor_matrix, axis=0)
        variance = np.var(self.sensor_matrix, axis=0)
        
        # Broadcasting: Instantly subtracts mean and divides by variance across all 12 numbers without a single loop.
        self.normalized_matrix = (self.sensor_matrix - mean) / np.sqrt(variance)
        
        print("\n--- 2. BROADCASTING & VARIANCE (Normalized Data) ---")
        print(np.round(self.normalized_matrix, 2))

        # PART 2: Linear Algebra 
        predictions = np.dot(self.normalized_matrix, self.weights)
        
        print("\n--- 3. LINEAR ALGEBRA (Initial Dot Product Predictions) ---")
        print(np.round(predictions, 2))

        # PART 2: Calculus (Gradients) - "How an AI Learns"
        error = predictions - self.labels
        initial_loss = np.mean(error ** 2)
        
        # 2. Calculus: The Gradient 
        learning_rate = 0.1
        gradients = np.dot(self.normalized_matrix.T, error) / len(self.labels)
        
        # 3. Update the weights
        self.weights = self.weights - (learning_rate * gradients)
        
        # 4. Prove the AI learned by re-calculating the error
        new_predictions = np.dot(self.normalized_matrix, self.weights)
        new_loss = np.mean((new_predictions - self.labels) ** 2)

        print("\n--- 4. CALCULUS (Gradient Descent Learning Step) ---")
        print(f"Initial Error (Loss): {initial_loss:.4f}")
        print(f"New Error (Loss) after Gradients: {new_loss:.4f}")
        print("Notice how the calculus successfully reduced the error!")

if __name__ == "__main__":
    engine = DroneNeuralEngine()
    engine.process_math()