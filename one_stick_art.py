from PIL import Image, ImageDraw
import random
import math
import matplotlib.pyplot as plt
import numpy as np

NB_STRING = 200# çizgi sayısı
NB_COORDINATES = 360  # koordinat sayısı
CANVAS_SIZE = 200  # resim büyüklüğü
POP_SIZE = 500  # popülasyon büyüklüğü
NB_GENERATION = 50  # toplam kaç generasyon ilerleneceği
STRING_WIDTH = 1  # çizilecek çizginin kalınlığı
MUTATION_RATE = 0.1  # mutasyon oranı

# Mutasyon burada gerçekleşir.
def mutation_individual(individual, mutation_rate):
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            # Her bir gen için x veya y koordinatlarından birini rastgele değiştirir
            gene = list(gene)
            gene[random.randint(0, 3)] = random.randint(0, CANVAS_SIZE)
            mutated_individual.append(tuple(gene))
        else:
            mutated_individual.append(gene)
    return mutated_individual

# Koordinatların rastgele üretilmesi
def generate_coordinates():
    coordinates = []
    for _ in range(NB_COORDINATES):
        angle = random.random() * math.pi * 2
        x = (math.cos(angle) * CANVAS_SIZE // 2) + CANVAS_SIZE // 2
        y = (math.sin(angle) * CANVAS_SIZE // 2) + CANVAS_SIZE // 2
        coordinates.append((x, y))
    return coordinates

# Rastgele birey oluşturulur
def create_random_individual(coordinates):
    individual = []
    for _ in range(NB_STRING):
        coord1, coord2 = random.sample(coordinates, 2)
        individual.append(coord1 + coord2)
    return individual

# Resmi bir diziye dönüştürme
def image_to_canvas(image):
    image = mask_circle_solid(image)
    image_pix = image.load()
    image_canvas = [[image_pix[x, y] for x in range(CANVAS_SIZE)] for y in range(CANVAS_SIZE)]
    return image_canvas

# Bireyden bir resim oluşturma
def create_image_from_individual(individual):
    img = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    img = mask_circle_solid(img)
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in individual:
        draw.line([(x1, y1), (x2, y2)], fill="black", width=STRING_WIDTH)
    return img

# İki resmi yan yana birleştirip kaydetme
def save_double_canvas(canvas1, canvas2, filename):
    res = []
    for r in canvas1:
        res.extend(map(tuple, r))
    im1 = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE))
    im1.putdata(res)

    res = []
    for r in canvas2:
        res.extend(map(tuple, r))
    im2 = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE))
    im2.putdata(res)

    im = Image.new('RGB', (CANVAS_SIZE * 2, CANVAS_SIZE))
    im.paste(im1, (0, 0))
    im.paste(im2, (CANVAS_SIZE, 0))
    im.save(filename)

# Resmi daire içine alma
def mask_circle_solid(image):
    background = Image.new(image.mode, image.size, (0, 0, 0))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)
    return Image.composite(image, background, mask)

# Fitness değeri hesaplama
def calculate_fitness(individual, image_canvas):
    image_individual = create_image_from_individual(individual)
    canvas_individual = image_to_canvas(image_individual)

    hamming_distance = 0
    for x in range(CANVAS_SIZE):
        for y in range(CANVAS_SIZE):
            if canvas_individual[x][y] != image_canvas[x][y]:
                hamming_distance += 1
    return hamming_distance

# 2 parenttan bir çocuk yapılır.
def reproduction_individual(individual_a, individual_b):
    rand = random.randrange(NB_STRING)
    child = individual_a[:rand] + individual_b[rand:]
    return child

# Function to calculate similarity percentage between two images
def calculate_similarity_percentage(image1, image2):
    # Convert images to numpy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Calculate percentage of pixels that are similar
    similarity_percentage = np.sum(arr1 == arr2) / arr1.size * 100
    return similarity_percentage

if __name__ == "__main__":
    image_name = "mickey"
    image = Image.open('images/'+image_name+'.png').convert('L').convert('RGB').resize((CANVAS_SIZE, CANVAS_SIZE))
    image_canvas = image_to_canvas(image)

    coordinates = generate_coordinates()  # Koordinatların üretilmesi

    population = [create_random_individual(coordinates) for _ in range(POP_SIZE)]
    hamming_distance = []  # Her neslin benzerlik değerlerinin saklanması
    similarity_percentages = []  # Benzerlik yüzdelerini tutar

    for generation in range(NB_GENERATION):
        print(f'generation {generation}')
        pop_fitness = []
        # Burada fitness fonksiyonu ile en iyi üyeler bulunur.
        for individual in population:
            pop_fitness.append((calculate_fitness(individual, image_canvas), random.random(), individual))
        pop_fitness.sort()
        pop_fitness = pop_fitness[:POP_SIZE // 2]
        best_individual = min(zip(pop_fitness, population))[1]
        image_individual = create_image_from_individual(best_individual)
        canvas_individual = image_to_canvas(image_individual)
        save_double_canvas(image_canvas, canvas_individual, f'results/{image_name}/generation{generation}.png')

        population = [popf[2] for popf in pop_fitness]
        children = []
        # POP_SIZE tam olana kadar çocuk yapma
        while len(children) + len(population) < POP_SIZE:
            parent1, parent2 = random.sample(population, 2)
            child = reproduction_individual(parent1, parent2)
            children.append(child)

        population.extend(children)

        # Hamming distanceları diziye alma
        total_hamming_distance = sum(fit for fit, _, _ in pop_fitness)
        hamming_distance.append(total_hamming_distance)

        # Calculate and store similarity percentage
        similarity_percentage = calculate_similarity_percentage(image, image_individual)
        similarity_percentages.append(similarity_percentage)

    # Benzerlik grafiği oluşturma
    plt.plot(range(NB_GENERATION), hamming_distance)
    plt.xlabel('Generation')
    plt.ylabel('Hamming Distance')
    plt.title('Similarity graph')
    plt.show()

    # Similarity percentage graph
    plt.plot(range(NB_GENERATION), similarity_percentages)
    plt.xlabel('Generation')
    plt.ylabel('Similarity Percentage')
    plt.title('Similarity Percentage with First Read Image')
    plt.show()
