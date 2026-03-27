def find_number(nums, target):
    for num in nums:
        if num == target:
            return True
    return False
print(find_number([10, 11, 12, 13], 13))
print(find_number([4], 7))