"""Реализация задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    if vectors.ndim == 2:
        vectors = vectors[..., np.newaxis]
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.asarray(data.matrix)
    threshold = data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    
    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)
    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    r_idx, c_idx = np.indices((rows, columns))
    mask = (r_idx + c_idx) % 2 == 0
    return np.where(mask, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color = np.array(data.shape_color, dtype=np.uint8)
    background_color = np.array(data.background_color, dtype=np.uint8)
    
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    
    cy, cx = image_height // 2, image_width // 2
    y_start = cy - height // 2
    y_end = y_start + height
    x_start = cx - width // 2
    x_end = x_start + width
    
    y_min = max(0, y_start)
    y_max = min(image_height, y_end)
    x_min = max(0, x_start)
    x_max = min(image_width, x_end)
    
    image[y_min:y_max, x_min:x_max] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color = np.array(data.shape_color, dtype=np.uint8)
    background_color = np.array(data.background_color, dtype=np.uint8)
    
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    
    y_indices, x_indices = np.ogrid[:image_height, :image_width]
    cy, cx = image_height // 2, image_width // 2
    
    if semi_axis_x > 0 and semi_axis_y > 0:
        mask = (((x_indices - cx) ** 2) / (semi_axis_x ** 2) + 
                ((y_indices - cy) ** 2) / (semi_axis_y ** 2)) <= 1.0
        image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    window = data.window
    
    mean_val = float(np.mean(values))
    var_val = float(np.var(values))
    std_val = float(np.std(values))
    
    local_maxima = []
    local_minima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)
            
    if window > 0 and len(values) >= window:
        weights = np.ones(window) / window
        moving_avg = np.convolve(values, weights, mode='valid')
    else:
        moving_avg = np.array([])
        
    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std=std_val,
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_avg
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1 if labels.size > 0 else 0
    
    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size > 0:
        result[np.arange(labels.size), labels] = 1
    return result




# тесты
if __name__ == "__main__":
    # 1
    print("Тестирование sum_prod...")
    mats = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    vecs = np.array([[[1], [2]], [[3], [4]]])
    # A1*v1 = [[1, 0], [0, 1]] @ [[1], [2]] = [[1], [2]]
    # A2*v2 = [[2, 0], [0, 2]] @ [[3], [4]] = [[6], [8]]
    # Сумма = [[7], [10]]
    res_sum_prod = sum_prod(MatrixVectorBatchInput(mats, vecs))
    np.testing.assert_array_equal(res_sum_prod, [[7], [10]])
    assert res_sum_prod.shape == (2, 1)

    # 2
    print("Тестирование binarize...")
    mat = np.array([[0.1, 0.6, 0.5], [0.8, 0.3, -1.0]])
    bin_res = binarize(BinarizeInput(mat, threshold=0.5))
    expected_bin = np.array([
        [0, 1, 0],
        [1, 0, 0]
    ])
    np.testing.assert_array_equal(bin_res, expected_bin)
    np.testing.assert_array_equal(mat, [[0.1, 0.6, 0.5], [0.8, 0.3, -1.0]])

    # 3
    print("Тестирование unique_rows и unique_columns...")
    mat_uniq = np.array([
        [1, 2, 2],
        [3, 3, 4]
    ])
    assert unique_rows(MatrixInput(mat_uniq)) == [[1.0, 2.0], [3.0, 4.0]]
    assert unique_columns(MatrixInput(mat_uniq)) == [[1.0, 3.0], [2.0, 3.0], [2.0, 4.0]]

    # 4
    print("Тестирование matrix_statistics...")
    stats = matrix_statistics(RandomMatrixInput(rows=4, columns=5, mean=0.0, std=1.0, seed=42))
    assert isinstance(stats, MatrixStatistics)
    assert stats.matrix.shape == (4, 5)
    assert stats.row_means.shape == (4,)
    assert stats.column_means.shape == (5,)
    assert stats.row_variances.shape == (4,)
    assert stats.column_variances.shape == (5,)

    # 5
    print("Тестирование chess...")
    chess_mat = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected_chess = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    np.testing.assert_array_equal(chess_mat, expected_chess)

    chess_odd = chess(ChessInput(rows=2, columns=2, first=5.5, second=9.9))
    expected_odd = np.array([
        [5.5, 9.9],
        [9.9, 5.5]
    ])
    np.testing.assert_array_equal(chess_odd, expected_odd)

    # 6
    print("Тестирование draw_rectangle и draw_ellipse...")
    rect = draw_rectangle(RectangleInput(
        width=2, height=2, image_height=5, image_width=5, 
        shape_color=(255, 0, 0), background_color=(0, 0, 0)
    ))
    assert rect.shape == (5, 5, 3)
    np.testing.assert_array_equal(rect[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(rect[2, 2], [255, 0, 0])

    ellipse = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=1, image_height=5, image_width=5, 
        shape_color=(0, 255, 0), background_color=(0, 0, 0)
    ))
    assert ellipse.shape == (5, 5, 3)
    np.testing.assert_array_equal(ellipse[2, 2], [0, 255, 0])

    ellipse_zero = draw_ellipse(EllipseInput(
        semi_axis_x=0, semi_axis_y=0, image_height=3, image_width=3, 
        shape_color=(255, 255, 255), background_color=(0, 0, 0)
    ))
    assert ellipse_zero.shape == (3, 3, 3)

    # 7
    print("Тестирование analyze_time_series...")
    ts_data = [1, 3, 2, 4, 1]
    ts_res = analyze_time_series(TimeSeriesInput(values=ts_data, window=3))
    
    assert ts_res.local_maxima_indices == [1, 3]  
    assert ts_res.local_minima_indices == [2] 
    assert abs(ts_res.mean - 2.2) < 1e-5
    np.testing.assert_allclose(ts_res.moving_average, [2.0, 3.0, 2.3333333333333335], atol=1e-5)

    # 8
    print("Тестирование one_hot...")
    oh = one_hot(OneHotInput(labels=[0, 2, 3, 0]))
    expected_oh = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    np.testing.assert_array_equal(oh, expected_oh)

    oh_explicit = one_hot(OneHotInput(labels=[1, 2], class_count=5))
    expected_explicit = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ])
    np.testing.assert_array_equal(oh_explicit, expected_explicit)

    print("Часть 2: все тесты успешно пройдены")