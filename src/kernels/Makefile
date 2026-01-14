CC			:=	nvcc

CFLAGS		:=
NVFLAGS 	:=
LDFLAGS		:=

TARGET		:= test_kernels

CPP_SRCS	:= $(wildcard *.cpp) $(wildcard ../shared/*.cpp)
CPP_OBJS	:= $(CPP_SRCS:.cpp=.o)

CU_SRCS 	:= $(wildcard *.cu)
CU_OBJS 	:= $(CU_SRCS:.cu=.o)

OBJS		:= $(CPP_OBJS) $(CU_OBJS)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS)
