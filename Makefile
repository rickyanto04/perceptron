# variabili del makefile
CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm
TARGET = nn_image_recognizer

#file sorgenti
SRCS = src/main.c src/neural_math.c src/network.c
#file oggetto (.o generati dalla prima compilazione)
OBJS = $(SRCS:.c=.o)

#regola principale, quella eseguita digitando 'make'
all: $(TARGET)

#regola di costruzione dell'eseguibile finale partendo dai .o
$(TARGET): $(OBJS)
		$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

#regola di compilazione di un .c in un .o
%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

#regola per pulizia dei file generati (eseguibili e .o)
clean:
		rm -f src/*.o $(TARGET)