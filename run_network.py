import pygame
import numpy as np
from PIL import Image
from convert_to_mnist import imageprepare
from preload_network import net, increment, test_data


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img


dim = (800, 400)
WIDTH, HEIGHT = dim

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode(dim)
clock = pygame.time.Clock()
running = True

myfont = pygame.font.SysFont('Monospace', 30)

idx = increment()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            idx = increment()

    img = gen_image(test_data[idx][0])
    img.save('./images/current.png', 'PNG')

    img = pygame.image.load('./images/current.png')
    img = pygame.transform.scale(img, (WIDTH / 2, HEIGHT))
    imgrect = img.get_rect()

    screen.fill(pygame.Color(0, 0, 0))
    screen.blit(img, imgrect)

    current = test_data[idx][0]

    img_number = test_data[idx][1]
    textsurface = myfont.render(
        'Real Number: ' + str(img_number), False, (255, 255, 255))
    screen.blit(textsurface, (WIDTH / 2, HEIGHT / 3))

    textsurface = myfont.render(
        'What AI thinks: ' + str(net.test((current, img_number))[1]), False, (255, 255, 255))
    screen.blit(textsurface, (WIDTH / 2, HEIGHT * (2 / 3)))

    pygame.display.flip()

    clock.tick(60)
