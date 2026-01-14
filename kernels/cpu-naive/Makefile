CC		:=	gcc
CFLAGS	:=
LDFLAGS	:=

TARGET := test_kernel

SRCS	:= test_kernel.c cpu_naive.c ../../shared/matrix_utils.c
OBJS	:= $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS)
