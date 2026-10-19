from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    return sum(1 for char in text if char in 'aeiou')


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    return bin(data.value).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if not predicted:
        return 0.0
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    factors = {}
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    result = []
    for p in sorted(factors.keys()):
        power = factors[p]
        if power == 1:
            result.append(f"({p})")
        else:
            result.append(f"({p}**{power})")
    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    n = data.value
    current_sum = 0
    k = 0
    while current_sum < n:
        k += 1
        current_sum += k * k
    if current_sum == n:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s = str(data.value)
    length = len(s)
    mid = length // 2
    
    if length % 2 == 1:
        left_part = s[:mid]
        right_part = s[mid + 1:]
    else:
        left_part = s[:mid - 1]
        right_part = s[mid + 1:]
        
    sum_left = sum(int(c) for c in left_part)
    sum_right = sum(int(c) for c in right_part)
    return sum_left == sum_right


# Тесты 

if __name__ == "__main__":
    
    # 1
    print("Тестирование count_vowels...")
    assert count_vowels(TextInput("Hello World")) == 3           
    assert count_vowels(TextInput("AEIOUaeiou")) == 10          
    assert count_vowels(TextInput("bcdfgh")) == 0               
    assert count_vowels(TextInput("")) == 0                     
    assert count_vowels(TextInput("Python 3.10!")) == 1     
    assert count_vowels(TextInput("a")) == 1                     

    # 2
    print("Тестирование has_unique_characters...")
    assert has_unique_characters(TextInput("abcdef")) is True    
    assert has_unique_characters(TextInput("hello")) is False  
    assert has_unique_characters(TextInput("a")) is True         
    assert has_unique_characters(TextInput("")) is True          
    assert has_unique_characters(TextInput("aba")) is False      
    assert has_unique_characters(TextInput("AbCa")) is True    

    # 3
    print("Тестирование count_one_bits...")
    assert count_one_bits(PositiveIntegerInput(5)) == 2         
    assert count_one_bits(PositiveIntegerInput(7)) == 3         
    assert count_one_bits(PositiveIntegerInput(1)) == 1        
    assert count_one_bits(PositiveIntegerInput(2)) == 1         
    assert count_one_bits(PositiveIntegerInput(8)) == 1          
    assert count_one_bits(PositiveIntegerInput(15)) == 4         

    # 4
    print("Тестирование multiplicative_persistence...")
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3   
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0   
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4 
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1  
    assert multiplicative_persistence(PositiveIntegerInput(25)) == 2   

    # 5
    print("Тестирование mse...")
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0          
    assert mse(VectorPairInput([1, 2, 3], [2, 3, 4])) == 1.0          
    assert mse(VectorPairInput([], [])) == 0.0                         
    assert mse(VectorPairInput([-1, -2], [1, 2])) == 10.0             
    assert mse(VectorPairInput([0, 0], [0, 0])) == 0.0               

    # 6
    print("Тестирование prime_factorization...")
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"       
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"    
    assert prime_factorization(PositiveIntegerInput(4)) == "(2**2)"    
    assert prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)" 

    # 7
    print("Тестирование pyramid...")
    assert pyramid(PositiveIntegerInput(5)) == 2          
    assert pyramid(PositiveIntegerInput(1)) == 1          
    assert pyramid(PositiveIntegerInput(14)) == 3          
    assert pyramid(PositiveIntegerInput(3)) == "It is impossible"  
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"  

    # 8
    print("Тестирование is_balanced_number...")
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True   
    assert is_balanced_number(PositiveIntegerInput(123456)) is False  
    assert is_balanced_number(PositiveIntegerInput(11)) is True       
    assert is_balanced_number(PositiveIntegerInput(121)) is True      
    assert is_balanced_number(PositiveIntegerInput(5)) is True         
    assert is_balanced_number(PositiveIntegerInput(123)) is False      

    print(" Все расширенные тесты для Части 1 успешно пройдены!")