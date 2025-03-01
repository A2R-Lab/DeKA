# Online Parameter Estimation Using the Kaczmarz Method

### Overview
This project implements an **online parameter estimation algorithm** leveraging the **Kaczmarz method** to address overdetermined system identification problems efficiently. Our algorithm is compared with traditional 
parameter estimation methods like RLS and Kalman Filter.

### Key Features
- **Kaczmarz Method**: A lightweight iterative method that uses incoming data points to continuously refine parameter estimates.
- **Online Adaptability**: The method operates in real-time, updating estimates as new data becomes available.
- **Overdetermined Systems**: Efficiently handles systems with more equations than unknowns, leveraging redundancy for improved accuracy.

### Potential Impact
- Improved system control through real-time parameter updates.
- Reduced computational overhead compared to traditional recursive/batch methods.

### Status
- **Current Stage**: Algorithm development and testing in simulation.
- **Next Steps**: Expanding use cases to include physical robotic systems and preparing for submission to RSS 2025.
