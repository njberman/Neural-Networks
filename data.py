from numpy import array, float32

training_data = [(array([[0.], [0.]], dtype=float32), array([[0.], [1.]])), (array([[1.], [0.]], dtype=float32), array(
    [[0.], [1.]])), (array([[0.], [1.]], dtype=float32), array([[0.], [1.]])), (array([[1.], [1.]], dtype=float32), array([[1.], [1.]]))]

test_data = [(array([[0.], [0.]], dtype=float32), 0), (array([[0.], [1.]], dtype=float32), 0),
             (array([[1.], [1.]], dtype=float32), 1), (array([[1.], [0.]], dtype=float32), 0)]
