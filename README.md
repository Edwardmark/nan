This project is about tensorflow implementation for paper attention-based template adaptation for face verification

2017/5/23 
start project, make an overall plan and design the algorithm
First, I decide to train a neural network end-to-end to do video face verification rather than separately as NAN paper.
My goal is to set up the new state-of-the-art result in IJB-A datasets.

Today's goal:
Design the neural network architecture based on mobilenet, which is light and good.

Today's review:
Get mobilenet inference written down.
Found that with zero weight-decay, the mobilenet performance is relatively good.
But with 1e-4 decay, the performance degenerate significantly. 

Todo:
Run the mobilenet in VGG-Face or WebFace.

2017/05/25
inference not run correctly, need bug fix!