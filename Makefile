PROGNAME := hello_cuda_world

SRCDIR := src
OBJDIR := obj
SRCS := $(SRCDIR)/hello_cuda_world.cu

#######################################################
#以下おまじない

NVCC := nvcc
DUMP := cuobjdump --dump-sass
NVCFLAGS := -O3 -gencode arch=compute_53,code=sm_53
NVCINFFLAGS := --ptxas-options=-v
NVCPTXFLAGS := --ptx
NVCFATFLAGS := --fatbin
LDFLAGS :=
LIBS :=

PROG := $(OBJDIR)/$(PROGNAME)
OBJS := $(SRCS:%.cu=$(OBJDIR)/%.o)
DEPS := $(SRCS:%.cu=$(OBJDIR)/%.d)
PTXS := $(SRCS:%.cu=$(OBJDIR)/%.ptx)
FATS := $(SRCS:%.cu=$(OBJDIR)/%.fatbin)

.PHONY: all clean

all: $(PROG) $(PTXS) $(FATS)

$(PROG): $(OBJS)
		$(NVCC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.d:%.cu
		@mkdir -p $(OBJDIR)
		@mkdir -p $(OBJDIR)/$(SRCDIR)
#		@$(NVCC) -M $(NVCFLAGS) $< > $@
		@$(NVCC) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(NVCFLAGS) -c $< -o $@
		@sed -i '1s/^/$(OBJDIR)\/$(SRCDIR)\//' $@

-include $(DEPS)

$(OBJDIR)/%.o:%.cu
		@mkdir -p $(OBJDIR)
		@mkdir -p $(OBJDIR)/$(SRCDIR)
		$(NVCC) $(NVCFLAGS) $(NVCINFFLAGS) -c $< -o $@

$(OBJDIR)/%.ptx:%.cu
		@mkdir -p $(OBJDIR)
		@mkdir -p $(OBJDIR)/$(SRCDIR)
		$(NVCC) $(NVCFLAGS) $(NVCPTXFLAGS) -c $< -o $@

$(OBJDIR)/%.fatbin:%.cu
		@mkdir -p $(OBJDIR)
		@mkdir -p $(OBJDIR)/$(SRCDIR)
		$(NVCC) $(NVCFLAGS) $(NVCFATFLAGS) -c $< -o $@
		$(DUMP) $@ > $@.lst

clean:
		rm -rf $(OBJDIR)/*
