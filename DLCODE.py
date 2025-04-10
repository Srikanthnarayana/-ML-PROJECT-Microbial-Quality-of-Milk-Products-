import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import seaborn as sns

class MilkQualityAnalyzer:
    def __init__(self):
        self.model = None
        self.image = None
        self.processed_image = None
        self.color_features = None
        self.prediction = None
        self.threshold_values = {
            'pH': {'good': (6.6, 6.8), 'acceptable': (6.4, 7.0), 'poor': (0, 7.5)},
            'bacterial_count': {'good': (0, 100000), 'acceptable': (100000, 500000), 'poor': (500000, float('inf'))},
            'somatic_cell_count': {'good': (0, 200000), 'acceptable': (200000, 400000), 'poor': (400000, float('inf'))}
        }
        
    def load_image(self, image_path):
        """Load and preprocess the image of milk sample"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not load image from the provided path")
        
        # Convert to RGB for display purposes
        self.display_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        self.processed_image = self.preprocess_image(self.image)
        
        # Extract color features
        self.color_features = self.extract_color_features(self.processed_image)
        
        return self.display_image
    
    def preprocess_image(self, image):
        """Preprocess the image for analysis"""
        # Resize for consistency
        resized = cv2.resize(image, (300, 300))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        # Convert to different color spaces for feature extraction
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        
        return {'rgb': resized, 'hsv': hsv, 'lab': lab}
    
    def extract_color_features(self, processed_image):
        """Extract color features from the processed image"""
        features = {}
        
        # RGB features
        rgb = processed_image['rgb']
        features['mean_r'] = np.mean(rgb[:,:,2])
        features['mean_g'] = np.mean(rgb[:,:,1])
        features['mean_b'] = np.mean(rgb[:,:,0])
        features['std_r'] = np.std(rgb[:,:,2])
        features['std_g'] = np.std(rgb[:,:,1])
        features['std_b'] = np.std(rgb[:,:,0])
        
        # HSV features
        hsv = processed_image['hsv']
        features['mean_h'] = np.mean(hsv[:,:,0])
        features['mean_s'] = np.mean(hsv[:,:,1])
        features['mean_v'] = np.mean(hsv[:,:,2])
        features['std_h'] = np.std(hsv[:,:,0])
        features['std_s'] = np.std(hsv[:,:,1])
        features['std_v'] = np.std(hsv[:,:,2])
        
        # LAB features
        lab = processed_image['lab']
        features['mean_l'] = np.mean(lab[:,:,0])
        features['mean_a'] = np.mean(lab[:,:,1])
        features['mean_b_lab'] = np.mean(lab[:,:,2])
        features['std_l'] = np.std(lab[:,:,0])
        features['std_a'] = np.std(lab[:,:,1])
        features['std_b_lab'] = np.std(lab[:,:,2])
        
        # Calculate whiteness index
        features['whiteness'] = (features['mean_r'] + features['mean_g'] + features['mean_b']) / 3
        
        # Calculate yellowness index (higher in spoiled milk)
        features['yellowness'] = features['mean_r'] - features['mean_b']
        
        return features
    
    def colorimetric_analysis(self):
        """Perform colorimetric analysis based on extracted features"""
        if self.color_features is None:
            raise ValueError("No image has been loaded and processed")
        
        # Simple colorimetric rules for milk quality
        # These thresholds would need to be calibrated with known samples
        whiteness = self.color_features['whiteness']
        yellowness = self.color_features['yellowness']
        saturation = self.color_features['mean_s']
        
        # Determine quality based on colorimetric features
        if whiteness > 200 and yellowness < 15 and saturation < 30:
            quality = "Good"
            confidence = min(100, (whiteness - 180) / 0.75)
        elif whiteness > 180 and yellowness < 25 and saturation < 50:
            quality = "Acceptable"
            confidence = min(100, (200 - whiteness) / 0.2 + 50)
        else:
            quality = "Poor"
            confidence = min(100, (255 - whiteness) / 0.75 + yellowness)
        
        return {
            'quality': quality,
            'confidence': confidence,
            'whiteness': whiteness,
            'yellowness': yellowness,
            'saturation': saturation
        }
    
    def train_model(self, dataset_path):
        """Train a machine learning model on milk quality dataset"""
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Prepare features and target
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'model': self.model
        }
    
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model has been trained")
        
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def predict_quality(self):
        """Predict milk quality using the trained model"""
        if self.model is None:
            raise ValueError("No model has been loaded or trained")
        
        if self.color_features is None:
            raise ValueError("No image has been loaded and processed")
        
        # Convert features to DataFrame with correct column order
        features_df = pd.DataFrame([self.color_features])
        
        # Make prediction
        self.prediction = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Get the probability of the predicted class
        pred_idx = list(self.model.classes_).index(self.prediction)
        confidence = probabilities[pred_idx] * 100
        
        return {
            'quality': self.prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def generate_report(self):
        """Generate a comprehensive report of the analysis"""
        if self.color_features is None:
            raise ValueError("No image has been loaded and processed")
        
        # Get colorimetric analysis
        colorimetric_result = self.colorimetric_analysis()
        
        # Get AI prediction if model is available
        ai_result = None
        if self.model is not None:
            ai_result = self.predict_quality()
        
        # Combine results
        report = {
            'colorimetric_analysis': colorimetric_result,
            'ai_prediction': ai_result,
            'color_features': self.color_features
        }
        
        return report
    
    def visualize_results(self, report, save_path=None):
        """Visualize the analysis results"""
        plt.figure(figsize=(15, 10))
        
        # Plot original image
        plt.subplot(2, 3, 1)
        plt.imshow(self.display_image)
        plt.title('Original Milk Sample')
        plt.axis('off')
        
        # Plot color distribution
        plt.subplot(2, 3, 2)
        color_data = [
            report['color_features']['mean_r'],
            report['color_features']['mean_g'],
            report['color_features']['mean_b']
        ]
        plt.bar(['Red', 'Green', 'Blue'], color_data, color=['r', 'g', 'b'])
        plt.title('RGB Color Distribution')
        plt.ylim(0, 255)
        
        # Plot colorimetric analysis
        plt.subplot(2, 3, 3)
        colorimetric = report['colorimetric_analysis']
        metrics = ['Whiteness', 'Yellowness', 'Saturation']
        values = [colorimetric['whiteness'], colorimetric['yellowness'], colorimetric['saturation']]
        plt.bar(metrics, values, color=['lightblue', 'yellow', 'purple'])
        plt.title('Colorimetric Metrics')
        
        # Plot quality assessment
        plt.subplot(2, 3, 4)
        quality_label = colorimetric['quality']
        confidence = colorimetric['confidence']
        plt.text(0.5, 0.5, f"Quality: {quality_label}\nConfidence: {confidence:.2f}%",
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.title('Colorimetric Assessment')
        
        # Plot AI prediction if available
        plt.subplot(2, 3, 5)
        if report['ai_prediction']:
            ai_quality = report['ai_prediction']['quality']
            ai_confidence = report['ai_prediction']['confidence']
            probabilities = report['ai_prediction']['probabilities']
            
            # Create bar chart of probabilities
            qualities = list(probabilities.keys())
            probs = list(probabilities.values())
            colors = ['green' if q == 'good' else 'orange' if q == 'acceptable' else 'red' for q in qualities]
            plt.bar(qualities, probs, color=colors)
            plt.title(f'AI Prediction: {ai_quality} ({ai_confidence:.2f}%)')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, "AI model not available",
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.title('AI Prediction')
        
        # Plot comparison if both methods are available
        plt.subplot(2, 3, 6)
        if report['ai_prediction']:
            labels = ['Colorimetric', 'AI-based']
            methods = [colorimetric['quality'], ai_quality]
            colors = ['blue', 'purple']
            plt.bar(labels, [1, 1], color=colors)
            plt.text(0, 0.5, f"{colorimetric['quality']}\n({colorimetric['confidence']:.2f}%)",
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=10, fontweight='bold')
            plt.text(1, 0.5, f"{ai_quality}\n({ai_confidence:.2f}%)",
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=10, fontweight='bold')
            plt.title('Method Comparison')
            plt.ylim(0, 2)
        else:
            plt.text(0.5, 0.5, "Comparison not available",
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.title('Method Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Results visualization saved to {save_path}")
        
        plt.show()


class MilkQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Milk Quality Analyzer")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        self.analyzer = MilkQualityAnalyzer()
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Rapid Colorimetric and AI-Based Milk Quality Analyzer", 
                              font=("Arial", 18, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)
        
        # Image frame
        self.image_frame = tk.Frame(main_frame, bg="white", bd=2, relief=tk.GROOVE)
        self.image_frame.pack(pady=10)
        
        # Default image display
        self.image_label = tk.Label(self.image_frame, text="No image loaded", 
                                   width=50, height=20, bg="white")
        self.image_label.pack()
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        # Load image button
        load_btn = tk.Button(button_frame, text="Load Milk Sample Image", 
                            command=self.load_image, bg="#4CAF50", fg="white",
                            font=("Arial", 12), padx=10, pady=5)
        load_btn.grid(row=0, column=0, padx=10)
        
        # Analyze button
        analyze_btn = tk.Button(button_frame, text="Analyze Sample", 
                               command=self.analyze_sample, bg="#2196F3", fg="white",
                               font=("Arial", 12), padx=10, pady=5)
        analyze_btn.grid(row=0, column=1, padx=10)
        
        # Load model button
        load_model_btn = tk.Button(button_frame, text="Load AI Model", 
                                  command=self.load_model, bg="#9C27B0", fg="white",
                                  font=("Arial", 12), padx=10, pady=5)
        load_model_btn.grid(row=0, column=2, padx=10)
        
        # Train model button
        train_model_btn = tk.Button(button_frame, text="Train New Model", 
                                   command=self.train_model, bg="#FF9800", fg="white",
                                   font=("Arial", 12), padx=10, pady=5)
        train_model_btn.grid(row=0, column=3, padx=10)
        
        # Results frame
        self.results_frame = tk.Frame(main_frame, bg="white", bd=2, relief=tk.GROOVE)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results label
        self.results_label = tk.Label(self.results_frame, text="Analysis results will appear here", 
                                     bg="white", font=("Arial", 12), justify=tk.LEFT)
        self.results_label.pack(padx=20, pady=20)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Milk Sample Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading image...")
                self.root.update()
                
                # Load and display image
                display_image = self.analyzer.load_image(file_path)
                self.display_image(display_image)
                
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def display_image(self, image):
        # Resize image for display
        h, w = image.shape[:2]
        max_size = 400
        
        if h > max_size or w > max_size:
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
            
            image = cv2.resize(image, (new_w, new_h))
        
        # Convert to PIL format
        pil_image = Image.fromarray(image)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update image label
        self.image_label.config(image=tk_image, width=pil_image.width, height=pil_image.height)
        self.image_label.image = tk_image  # Keep a reference
    
    def analyze_sample(self):
        if self.analyzer.color_features is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Analyzing sample...")
            self.root.update()
            
            # Generate report
            report = self.analyzer.generate_report()
            
            # Display results
            self.display_results(report)
            
            # Visualize results
            self.analyzer.visualize_results(report)
            
            self.status_var.set("Analysis complete")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")
    
    def display_results(self, report):
        # Format results text
        colorimetric = report['colorimetric_analysis']
        
        results_text = f"COLORIMETRIC ANALYSIS:\n"
        results_text += f"Quality: {colorimetric['quality']}\n"
        results_text += f"Confidence: {colorimetric['confidence']:.2f}%\n\n"
        results_text += f"Whiteness: {colorimetric['whiteness']:.2f}\n"
        results_text += f"Yellowness: {colorimetric['yellowness']:.2f}\n"
        results_text += f"Saturation: {colorimetric['saturation']:.2f}\n\n"
        
        if report['ai_prediction']:
            ai_result = report['ai_prediction']
            results_text += f"AI-BASED ANALYSIS:\n"
            results_text += f"Predicted Quality: {ai_result['quality']}\n"
            results_text += f"Confidence: {ai_result['confidence']:.2f}%\n\n"
            
            results_text += "Class Probabilities:\n"
            for quality, prob in ai_result['probabilities'].items():
                results_text += f"- {quality}: {prob*100:.2f}%\n"
        else:
            results_text += "AI-BASED ANALYSIS:\n"
            results_text += "No AI model loaded\n"
        
        # Update results label
        self.results_label.config(text=results_text)
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select AI Model File",
            filetypes=[("Model files", "*.pkl *.joblib")]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading model...")
                self.root.update()
                
                self.analyzer.load_model(file_path)
                
                self.status_var.set(f"Model loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "AI model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.status_var.set("Error loading model")
    
    def train_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Training Dataset",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                self.status_var.set("Training model...")
                self.root.update()
                
                # Train model
                results = self.analyzer.train_model(file_path)
                
                # Show training results
                accuracy = results['accuracy'] * 100
                messagebox.showinfo("Training Complete", 
                                   f"Model trained successfully!\nAccuracy: {accuracy:.2f}%")
                
                # Ask to save model
                save = messagebox.askyesno("Save Model", 
                                          "Do you want to save the trained model?")
                if save:
                    save_path = filedialog.asksaveasfilename(
                        title="Save Model",
                        defaultextension=".joblib",
                        filetypes=[("Joblib files", "*.joblib")]
                    )
                    if save_path:
                        self.analyzer.save_model(save_path)
                
                self.status_var.set("Model training complete")
            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {str(e)}")
                self.status_var.set("Training failed")


# Generate synthetic dataset for demonstration
# Fix the generate_synthetic_dataset function to use pandas concat instead of append
def generate_synthetic_dataset(num_samples=100, output_path=None):
    np.random.seed(42)
    
    # Create empty dataframe
    columns = [
        'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
        'mean_h', 'mean_s', 'mean_v', 'std_h', 'std_s', 'std_v',
        'mean_l', 'mean_a', 'mean_b_lab', 'std_l', 'std_a', 'std_b_lab',
        'whiteness', 'yellowness', 'quality'
    ]
    df = pd.DataFrame(columns=columns)
    all_samples = []
    
    # Generate good quality samples
    good_samples = int(num_samples * 0.4)
    for i in range(good_samples):
        sample = {
            'mean_r': np.random.uniform(200, 240),
            'mean_g': np.random.uniform(200, 240),
            'mean_b': np.random.uniform(200, 240),
            'std_r': np.random.uniform(5, 15),
            'std_g': np.random.uniform(5, 15),
            'std_b': np.random.uniform(5, 15),
            'mean_h': np.random.uniform(20, 40),
            'mean_s': np.random.uniform(5, 20),
            'mean_v': np.random.uniform(200, 240),
            'std_h': np.random.uniform(2, 8),
            'std_s': np.random.uniform(2, 8),
            'std_v': np.random.uniform(5, 15),
            'mean_l': np.random.uniform(200, 240),
            'mean_a': np.random.uniform(120, 130),
            'mean_b_lab': np.random.uniform(120, 130),
            'std_l': np.random.uniform(5, 15),
            'std_a': np.random.uniform(2, 8),
            'std_b_lab': np.random.uniform(2, 8),
            'quality': 'good'
        }
        # Calculate derived features
        sample['whiteness'] = (sample['mean_r'] + sample['mean_g'] + sample['mean_b']) / 3
        sample['yellowness'] = sample['mean_r'] - sample['mean_b']
        
        all_samples.append(sample)
    
    # Generate acceptable quality samples
    acceptable_samples = int(num_samples * 0.3)
    for i in range(acceptable_samples):
        sample = {
            'mean_r': np.random.uniform(180, 220),
            'mean_g': np.random.uniform(180, 220),
            'mean_b': np.random.uniform(170, 210),
            'std_r': np.random.uniform(10, 20),
            'std_g': np.random.uniform(10, 20),
            'std_b': np.random.uniform(10, 20),
            'mean_h': np.random.uniform(30, 50),
            'mean_s': np.random.uniform(15, 35),
            'mean_v': np.random.uniform(180, 220),
            'std_h': np.random.uniform(5, 15),
            'std_s': np.random.uniform(5, 15),
            'std_v': np.random.uniform(10, 20),
            'mean_l': np.random.uniform(180, 220),
            'mean_a': np.random.uniform(125, 135),
            'mean_b_lab': np.random.uniform(125, 135),
            'std_l': np.random.uniform(10, 20),
            'std_a': np.random.uniform(5, 15),
            'std_b_lab': np.random.uniform(5, 15),
            'quality': 'acceptable'
        }
        # Calculate derived features
        sample['whiteness'] = (sample['mean_r'] + sample['mean_g'] + sample['mean_b']) / 3
        sample['yellowness'] = sample['mean_r'] - sample['mean_b']
        
        all_samples.append(sample)
    
    # Generate poor quality samples
    poor_samples = num_samples - good_samples - acceptable_samples
    for i in range(poor_samples):
        sample = {
            'mean_r': np.random.uniform(150, 200),
            'mean_g': np.random.uniform(140, 190),
            'mean_b': np.random.uniform(120, 170),
            'std_r': np.random.uniform(15, 30),
            'std_g': np.random.uniform(15, 30),
            'std_b': np.random.uniform(15, 30),
            'mean_h': np.random.uniform(40, 70),
            'mean_s': np.random.uniform(30, 60),
            'mean_v': np.random.uniform(150, 200),
            'std_h': np.random.uniform(10, 25),
            'std_s': np.random.uniform(10, 25),
            'std_v': np.random.uniform(15, 30),
            'mean_l': np.random.uniform(150, 200),
            'mean_a': np.random.uniform(130, 150),
            'mean_b_lab': np.random.uniform(130, 150),
            'std_l': np.random.uniform(15, 30),
            'std_a': np.random.uniform(10, 25),
            'std_b_lab': np.random.uniform(10, 25),
            'quality': 'poor'
        }
        # Calculate derived features
        sample['whiteness'] = (sample['mean_r'] + sample['mean_g'] + sample['mean_b']) / 3
        sample['yellowness'] = sample['mean_r'] - sample['mean_b']
        
        all_samples.append(sample)
    
    # Create dataframe from all samples
    df = pd.DataFrame(all_samples)
    
    # Save dataset if path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Synthetic dataset saved to {output_path}")
    
    return df

# ... existing code ...

# ... existing code ...

# ... existing code ...

if __name__ == "__main__":
    # Generate synthetic dataset
    # Make sure the directory exists before saving the file
    dataset_dir = "C:/Users/srikanth narayana/Desktop/dl/Data"
    
    # Only create directory if it doesn't exist already
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    dataset_path = os.path.join(dataset_dir, "milk_quality_dataset.csv")
    generate_synthetic_dataset(200, dataset_path)
    
    # Check if display is available before starting GUI
    try:
        # Start the application
        root = tk.Tk()
        app = MilkQualityApp(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"Error starting GUI: {e}")
        print("This application requires a graphical display.")
        print("If you're running this in a terminal, make sure X11 forwarding is enabled.")
        print("You can still use the non-GUI components of this code by importing the MilkQualityAnalyzer class.")

        #python3 "C:\Users\srikanth narayana\Desktop\dl\DLCODE.py"