def convert_pixel_distance_to_meters(pixel_distance, reference_meters_height, reference_pixel_height):
    return (pixel_distance * reference_meters_height) / reference_pixel_height

def convert_meters_to_pixel_distance(meters, reference_meters_height, reference_pixel_height):
    return (meters * reference_pixel_height) / reference_meters_height