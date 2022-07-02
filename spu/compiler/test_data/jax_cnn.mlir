module @xla_computation_train_and_evaluate.6651 {
  func @main(%arg0: tensor<30x28x28x1x!pphlo.pub<f32>>, %arg1: tensor<30x!pphlo.pub<i32>>, %arg2: tensor<50x28x28x1x!pphlo.pub<f32>>, %arg3: tensor<50x!pphlo.pub<i32>>) -> tensor<!pphlo.pub<f32>> {
    %0 = "pphlo.constant"() {value = dense<2.000000e-02> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %1 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %2 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x10xf32>} : () -> tensor<50x10x!pphlo.pub<f32>>
    %3 = "pphlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<!pphlo.pub<f32>>
    %4 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<10xf32>} : () -> tensor<10x!pphlo.pub<f32>>
    %5 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %6 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x256xf32>} : () -> tensor<50x256x!pphlo.pub<f32>>
    %7 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<256xf32>} : () -> tensor<256x!pphlo.pub<f32>>
    %8 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %9 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<50x7x7x64xf32>} : () -> tensor<50x7x7x64x!pphlo.pub<f32>>
    %10 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x14x14x64xf32>} : () -> tensor<50x14x14x64x!pphlo.pub<f32>>
    %11 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<64xf32>} : () -> tensor<64x!pphlo.pub<f32>>
    %12 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %13 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<50x14x14x32xf32>} : () -> tensor<50x14x14x32x!pphlo.pub<f32>>
    %14 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<50x28x28x32xf32>} : () -> tensor<50x28x28x32x!pphlo.pub<f32>>
    %15 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<32xf32>} : () -> tensor<32x!pphlo.pub<f32>>
    %16 = "pphlo.constant"() {value = dense<-1.000000e-01> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %17 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x28x28x32xf32>} : () -> tensor<30x28x28x32x!pphlo.pub<f32>>
    %18 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<30x14x14x32xf32>} : () -> tensor<30x14x14x32x!pphlo.pub<f32>>
    %19 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x14x14x64xf32>} : () -> tensor<30x14x14x64x!pphlo.pub<f32>>
    %20 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<30x3136xf32>} : () -> tensor<30x3136x!pphlo.pub<f32>>
    %21 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x256xf32>} : () -> tensor<30x256x!pphlo.pub<f32>>
    %22 = "pphlo.constant"() {value = dense<0.0710529536> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %23 = "pphlo.constant"() {value = dense<1.99999988> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %24 = "pphlo.constant"() {value = dense<1.41421354> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %25 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %26 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %27 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %28 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %29 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %30 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %31 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %32 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %33 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %34 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %35 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %36 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %37 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %38 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %39 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %40 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %41 = "pphlo.constant"() {value = dense<1.00167406> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %42 = "pphlo.constant"() {value = dense<0.246640727> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %43 = "pphlo.constant"() {value = dense<2.83297682> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %44 = "pphlo.constant"() {value = dense<1.50140941> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %45 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %46 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %47 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %48 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<2560xf32>} : () -> tensor<2560x!pphlo.pub<f32>>
    %49 = "pphlo.constant"() {value = dense<1.90899944> : tensor<2560xf32>} : () -> tensor<2560x!pphlo.pub<f32>>
    %50 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<2560xf32>} : () -> tensor<2560x!pphlo.pub<f32>>
    %51 = "pphlo.constant"() {value = dense<1065353216> : tensor<2560xui32>} : () -> tensor<2560x!pphlo.pub<ui32>>
    %52 = "pphlo.constant"() {value = dense<9> : tensor<2560xui32>} : () -> tensor<2560x!pphlo.pub<ui32>>
    %53 = "pphlo.constant"() {value = dense<5> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %54 = "pphlo.constant"() {value = dense<26> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %55 = "pphlo.constant"() {value = dense<6> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %56 = "pphlo.constant"() {value = dense<17> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %57 = "pphlo.constant"() {value = dense<15> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %58 = "pphlo.constant"() {value = dense<19> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %59 = "pphlo.constant"() {value = dense<13> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %60 = "pphlo.constant"() {value = dense<4> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %61 = "pphlo.constant"() {value = dense<8> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %62 = "pphlo.constant"() {value = dense<24> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %63 = "pphlo.constant"() {value = dense<16> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %64 = "pphlo.constant"() {value = dense<3> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %65 = "pphlo.constant"() {value = dense<29> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %66 = "pphlo.constant"() {value = dense<2> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %67 = "pphlo.constant"() {value = dense<1> : tensor<1280xui32>} : () -> tensor<1280x!pphlo.pub<ui32>>
    %68 = "pphlo.constant"() {value = dense<466688986> : tensor<ui32>} : () -> tensor<!pphlo.pub<ui32>>
    %69 = "pphlo.constant"() {value = dense<5> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %70 = "pphlo.constant"() {value = dense<26> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %71 = "pphlo.constant"() {value = dense<6> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %72 = "pphlo.constant"() {value = dense<17> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %73 = "pphlo.constant"() {value = dense<15> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %74 = "pphlo.constant"() {value = dense<19> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %75 = "pphlo.constant"() {value = dense<13> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %76 = "pphlo.constant"() {value = dense<4> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %77 = "pphlo.constant"() {value = dense<8> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %78 = "pphlo.constant"() {value = dense<24> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %79 = "pphlo.constant"() {value = dense<16> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %80 = "pphlo.constant"() {value = dense<3> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %81 = "pphlo.constant"() {value = dense<29> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %82 = "pphlo.constant"() {value = dense<2> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %83 = "pphlo.constant"() {value = dense<1> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %84 = "pphlo.constant"() {value = dense<3995620053> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %85 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %86 = "pphlo.constant"() {value = dense<0.000000e+00> : tensor<30x10xf32>} : () -> tensor<30x10x!pphlo.pub<f32>>
    %87 = "pphlo.constant"() {value = dense<-0.0333333351> : tensor<30x10xf32>} : () -> tensor<30x10x!pphlo.pub<f32>>
    %88 = "pphlo.constant"() {value = dense<30> : tensor<30xi32>} : () -> tensor<30x!pphlo.pub<i32>>
    %89 = "pphlo.constant"() {value = dense<0> : tensor<30xi32>} : () -> tensor<30x!pphlo.pub<i32>>
    %90 = "pphlo.constant"() {value = dense<0.0203008428> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %91 = "pphlo.constant"() {value = dense<1.99999988> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %92 = "pphlo.constant"() {value = dense<1.41421354> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %93 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %94 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %95 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %96 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %97 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %98 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %99 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %100 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %101 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %102 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %103 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %104 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %105 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %106 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %107 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %108 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %109 = "pphlo.constant"() {value = dense<1.00167406> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %110 = "pphlo.constant"() {value = dense<0.246640727> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %111 = "pphlo.constant"() {value = dense<2.83297682> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %112 = "pphlo.constant"() {value = dense<1.50140941> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %113 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %114 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %115 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %116 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<802816xf32>} : () -> tensor<802816x!pphlo.pub<f32>>
    %117 = "pphlo.constant"() {value = dense<1.90899944> : tensor<802816xf32>} : () -> tensor<802816x!pphlo.pub<f32>>
    %118 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<802816xf32>} : () -> tensor<802816x!pphlo.pub<f32>>
    %119 = "pphlo.constant"() {value = dense<1065353216> : tensor<802816xui32>} : () -> tensor<802816x!pphlo.pub<ui32>>
    %120 = "pphlo.constant"() {value = dense<9> : tensor<802816xui32>} : () -> tensor<802816x!pphlo.pub<ui32>>
    %121 = "pphlo.constant"() {value = dense<5> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %122 = "pphlo.constant"() {value = dense<26> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %123 = "pphlo.constant"() {value = dense<6> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %124 = "pphlo.constant"() {value = dense<17> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %125 = "pphlo.constant"() {value = dense<15> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %126 = "pphlo.constant"() {value = dense<19> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %127 = "pphlo.constant"() {value = dense<13> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %128 = "pphlo.constant"() {value = dense<4> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %129 = "pphlo.constant"() {value = dense<8> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %130 = "pphlo.constant"() {value = dense<24> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %131 = "pphlo.constant"() {value = dense<16> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %132 = "pphlo.constant"() {value = dense<3> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %133 = "pphlo.constant"() {value = dense<29> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %134 = "pphlo.constant"() {value = dense<2> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %135 = "pphlo.constant"() {value = dense<1> : tensor<401408xui32>} : () -> tensor<401408x!pphlo.pub<ui32>>
    %136 = "pphlo.constant"() {value = dense<706584679> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %137 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %138 = "pphlo.constant"() {value = dense<2.500000e-01> : tensor<30x7x7x64xf32>} : () -> tensor<30x7x7x64x!pphlo.pub<f32>>
    %139 = "pphlo.constant"() {value = dense<0.0669893697> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %140 = "pphlo.constant"() {value = dense<1.99999988> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %141 = "pphlo.constant"() {value = dense<1.41421354> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %142 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %143 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %144 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %145 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %146 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %147 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %148 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %149 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %150 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %151 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %152 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %153 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %154 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %155 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %156 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %157 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %158 = "pphlo.constant"() {value = dense<1.00167406> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %159 = "pphlo.constant"() {value = dense<0.246640727> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %160 = "pphlo.constant"() {value = dense<2.83297682> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %161 = "pphlo.constant"() {value = dense<1.50140941> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %162 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %163 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %164 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %165 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<18432xf32>} : () -> tensor<18432x!pphlo.pub<f32>>
    %166 = "pphlo.constant"() {value = dense<1.90899944> : tensor<18432xf32>} : () -> tensor<18432x!pphlo.pub<f32>>
    %167 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<18432xf32>} : () -> tensor<18432x!pphlo.pub<f32>>
    %168 = "pphlo.constant"() {value = dense<1065353216> : tensor<18432xui32>} : () -> tensor<18432x!pphlo.pub<ui32>>
    %169 = "pphlo.constant"() {value = dense<9> : tensor<18432xui32>} : () -> tensor<18432x!pphlo.pub<ui32>>
    %170 = "pphlo.constant"() {value = dense<5> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %171 = "pphlo.constant"() {value = dense<26> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %172 = "pphlo.constant"() {value = dense<6> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %173 = "pphlo.constant"() {value = dense<17> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %174 = "pphlo.constant"() {value = dense<15> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %175 = "pphlo.constant"() {value = dense<19> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %176 = "pphlo.constant"() {value = dense<13> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %177 = "pphlo.constant"() {value = dense<4> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %178 = "pphlo.constant"() {value = dense<8> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %179 = "pphlo.constant"() {value = dense<24> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %180 = "pphlo.constant"() {value = dense<16> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %181 = "pphlo.constant"() {value = dense<3> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %182 = "pphlo.constant"() {value = dense<29> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %183 = "pphlo.constant"() {value = dense<2> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %184 = "pphlo.constant"() {value = dense<1> : tensor<9216xui32>} : () -> tensor<9216x!pphlo.pub<ui32>>
    %185 = "pphlo.constant"() {value = dense<2095399837> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %186 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %187 = "pphlo.constant"() {value = dense<5> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %188 = "pphlo.constant"() {value = dense<26> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %189 = "pphlo.constant"() {value = dense<6> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %190 = "pphlo.constant"() {value = dense<17> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %191 = "pphlo.constant"() {value = dense<15> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %192 = "pphlo.constant"() {value = dense<19> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %193 = "pphlo.constant"() {value = dense<13> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %194 = "pphlo.constant"() {value = dense<4> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %195 = "pphlo.constant"() {value = dense<8> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %196 = "pphlo.constant"() {value = dense<24> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %197 = "pphlo.constant"() {value = dense<16> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %198 = "pphlo.constant"() {value = dense<3> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %199 = "pphlo.constant"() {value = dense<29> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %200 = "pphlo.constant"() {value = dense<2> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %201 = "pphlo.constant"() {value = dense<1> : tensor<15xui32>} : () -> tensor<15x!pphlo.pub<ui32>>
    %202 = "pphlo.constant"() {value = dense<5> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %203 = "pphlo.constant"() {value = dense<26> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %204 = "pphlo.constant"() {value = dense<6> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %205 = "pphlo.constant"() {value = dense<17> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %206 = "pphlo.constant"() {value = dense<15> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %207 = "pphlo.constant"() {value = dense<19> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %208 = "pphlo.constant"() {value = dense<13> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %209 = "pphlo.constant"() {value = dense<4> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %210 = "pphlo.constant"() {value = dense<8> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %211 = "pphlo.constant"() {value = dense<24> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %212 = "pphlo.constant"() {value = dense<16> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %213 = "pphlo.constant"() {value = dense<3> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %214 = "pphlo.constant"() {value = dense<29> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %215 = "pphlo.constant"() {value = dense<2> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %216 = "pphlo.constant"() {value = dense<1> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %217 = "pphlo.constant"() {value = dense<0.378949106> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %218 = "pphlo.constant"() {value = dense<1.99999988> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %219 = "pphlo.constant"() {value = dense<1.41421354> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %220 = "pphlo.constant"() {value = dense<-3.000000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %221 = "pphlo.constant"() {value = dense<-2.500000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %222 = "pphlo.constant"() {value = dense<-2.00214257E-4> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %223 = "pphlo.constant"() {value = dense<2.81022636E-8> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %224 = "pphlo.constant"() {value = dense<1.00950558E-4> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %225 = "pphlo.constant"() {value = dense<3.43273939E-7> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %226 = "pphlo.constant"() {value = dense<0.00134934322> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %227 = "pphlo.constant"() {value = dense<-3.5233877E-6> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %228 = "pphlo.constant"() {value = dense<-0.00367342844> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %229 = "pphlo.constant"() {value = dense<-4.39150654E-6> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %230 = "pphlo.constant"() {value = dense<0.00573950773> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %231 = "pphlo.constant"() {value = dense<2.1858087E-4> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %232 = "pphlo.constant"() {value = dense<-0.0076224613> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %233 = "pphlo.constant"() {value = dense<-0.00125372503> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %234 = "pphlo.constant"() {value = dense<0.00943887047> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %235 = "pphlo.constant"() {value = dense<-0.00417768164> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %236 = "pphlo.constant"() {value = dense<1.00167406> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %237 = "pphlo.constant"() {value = dense<0.246640727> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %238 = "pphlo.constant"() {value = dense<2.83297682> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %239 = "pphlo.constant"() {value = dense<1.50140941> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %240 = "pphlo.constant"() {value = dense<5.000000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %241 = "pphlo.constant"() {value = dense<0x7F800000> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %242 = "pphlo.constant"() {value = dense<1.000000e+00> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %243 = "pphlo.constant"() {value = dense<-0.954499721> : tensor<288xf32>} : () -> tensor<288x!pphlo.pub<f32>>
    %244 = "pphlo.constant"() {value = dense<1.90899944> : tensor<288xf32>} : () -> tensor<288x!pphlo.pub<f32>>
    %245 = "pphlo.constant"() {value = dense<-1.000000e+00> : tensor<288xf32>} : () -> tensor<288x!pphlo.pub<f32>>
    %246 = "pphlo.constant"() {value = dense<1065353216> : tensor<288xui32>} : () -> tensor<288x!pphlo.pub<ui32>>
    %247 = "pphlo.constant"() {value = dense<9> : tensor<288xui32>} : () -> tensor<288x!pphlo.pub<ui32>>
    %248 = "pphlo.constant"() {value = dense<5> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %249 = "pphlo.constant"() {value = dense<26> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %250 = "pphlo.constant"() {value = dense<6> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %251 = "pphlo.constant"() {value = dense<17> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %252 = "pphlo.constant"() {value = dense<15> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %253 = "pphlo.constant"() {value = dense<19> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %254 = "pphlo.constant"() {value = dense<13> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %255 = "pphlo.constant"() {value = dense<4> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %256 = "pphlo.constant"() {value = dense<8> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %257 = "pphlo.constant"() {value = dense<24> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %258 = "pphlo.constant"() {value = dense<16> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %259 = "pphlo.constant"() {value = dense<3> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %260 = "pphlo.constant"() {value = dense<29> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %261 = "pphlo.constant"() {value = dense<2> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %262 = "pphlo.constant"() {value = dense<1> : tensor<144xui32>} : () -> tensor<144x!pphlo.pub<ui32>>
    %263 = "pphlo.constant"() {value = dense<3798891600> : tensor<1xui32>} : () -> tensor<1x!pphlo.pub<ui32>>
    %264 = "pphlo.constant"() {value = dense<466688986> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %265 = "pphlo.constant"() {value = dense<466688990> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %266 = "pphlo.constant"() {value = dense<466688987> : tensor<2xui32>} : () -> tensor<2x!pphlo.pub<ui32>>
    %267 = "pphlo.constant"() {value = dense<-1.99999988> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %268 = "pphlo.broadcast"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pub<i32>>) -> tensor<50x10x!pphlo.pub<i32>>
    %269 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<10x!pphlo.pub<i32>>
    %270 = "pphlo.broadcast"(%269) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10x!pphlo.pub<i32>>) -> tensor<50x10x!pphlo.pub<i32>>
    %271 = "pphlo.equal"(%268, %270) : (tensor<50x10x!pphlo.pub<i32>>, tensor<50x10x!pphlo.pub<i32>>) -> tensor<50x10x!pphlo.pub<i1>>
    %272 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<288x!pphlo.pub<ui32>>
    %273 = "pphlo.slice"(%272) {limit_indices = dense<144> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<288x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %274 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4x!pphlo.pub<ui32>>
    %275 = "pphlo.slice"(%274) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %276 = "pphlo.slice"(%274) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %277 = "pphlo.add"(%275, %276) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %278 = "pphlo.shift_left"(%276, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %279 = "pphlo.shift_right_logical"(%276, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %280 = "pphlo.or"(%278, %279) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %281 = "pphlo.xor"(%277, %280) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %282 = "pphlo.add"(%277, %281) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %283 = "pphlo.shift_left"(%281, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %284 = "pphlo.shift_right_logical"(%281, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %285 = "pphlo.or"(%283, %284) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %286 = "pphlo.xor"(%282, %285) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %287 = "pphlo.add"(%282, %286) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %288 = "pphlo.shift_left"(%286, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %289 = "pphlo.shift_right_logical"(%286, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %290 = "pphlo.or"(%288, %289) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %291 = "pphlo.xor"(%287, %290) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %292 = "pphlo.add"(%287, %291) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %293 = "pphlo.shift_left"(%291, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %294 = "pphlo.shift_right_logical"(%291, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %295 = "pphlo.or"(%293, %294) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %296 = "pphlo.xor"(%292, %295) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %297 = "pphlo.add"(%296, %266) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %298 = "pphlo.add"(%292, %297) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %299 = "pphlo.shift_left"(%297, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %300 = "pphlo.shift_right_logical"(%297, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %301 = "pphlo.or"(%299, %300) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %302 = "pphlo.xor"(%298, %301) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %303 = "pphlo.add"(%298, %302) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %304 = "pphlo.shift_left"(%302, %214) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %305 = "pphlo.shift_right_logical"(%302, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %306 = "pphlo.or"(%304, %305) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %307 = "pphlo.xor"(%303, %306) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %308 = "pphlo.add"(%303, %307) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %309 = "pphlo.shift_left"(%307, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %310 = "pphlo.shift_right_logical"(%307, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %311 = "pphlo.or"(%309, %310) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %312 = "pphlo.xor"(%308, %311) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %313 = "pphlo.add"(%308, %312) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %314 = "pphlo.add"(%313, %264) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %315 = "pphlo.shift_left"(%312, %211) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %316 = "pphlo.shift_right_logical"(%312, %210) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %317 = "pphlo.or"(%315, %316) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %318 = "pphlo.xor"(%313, %317) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %319 = "pphlo.add"(%318, %215) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %320 = "pphlo.add"(%314, %319) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %321 = "pphlo.shift_left"(%319, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %322 = "pphlo.shift_right_logical"(%319, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %323 = "pphlo.or"(%321, %322) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %324 = "pphlo.xor"(%320, %323) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %325 = "pphlo.add"(%320, %324) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %326 = "pphlo.shift_left"(%324, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %327 = "pphlo.shift_right_logical"(%324, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %328 = "pphlo.or"(%326, %327) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %329 = "pphlo.xor"(%325, %328) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %330 = "pphlo.add"(%325, %329) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %331 = "pphlo.shift_left"(%329, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %332 = "pphlo.shift_right_logical"(%329, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %333 = "pphlo.or"(%331, %332) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %334 = "pphlo.xor"(%330, %333) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %335 = "pphlo.add"(%330, %334) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %336 = "pphlo.shift_left"(%334, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %337 = "pphlo.shift_right_logical"(%334, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %338 = "pphlo.or"(%336, %337) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %339 = "pphlo.xor"(%335, %338) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %340 = "pphlo.add"(%339, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %341 = "pphlo.add"(%335, %340) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %342 = "pphlo.shift_left"(%340, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %343 = "pphlo.shift_right_logical"(%340, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %344 = "pphlo.or"(%342, %343) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %345 = "pphlo.xor"(%341, %344) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %346 = "pphlo.add"(%341, %345) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %347 = "pphlo.shift_left"(%345, %214) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %348 = "pphlo.shift_right_logical"(%345, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %349 = "pphlo.or"(%347, %348) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %350 = "pphlo.xor"(%346, %349) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %351 = "pphlo.add"(%346, %350) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %352 = "pphlo.shift_left"(%350, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %353 = "pphlo.shift_right_logical"(%350, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %354 = "pphlo.or"(%352, %353) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %355 = "pphlo.xor"(%351, %354) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %356 = "pphlo.add"(%351, %355) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %357 = "pphlo.shift_left"(%355, %211) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %358 = "pphlo.shift_right_logical"(%355, %210) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %359 = "pphlo.or"(%357, %358) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %360 = "pphlo.xor"(%356, %359) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %361 = "pphlo.add"(%360, %265) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %362 = "pphlo.add"(%356, %361) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %363 = "pphlo.shift_left"(%361, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %364 = "pphlo.shift_right_logical"(%361, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %365 = "pphlo.or"(%363, %364) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %366 = "pphlo.xor"(%362, %365) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %367 = "pphlo.add"(%362, %366) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %368 = "pphlo.shift_left"(%366, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %369 = "pphlo.shift_right_logical"(%366, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %370 = "pphlo.or"(%368, %369) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %371 = "pphlo.xor"(%367, %370) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %372 = "pphlo.add"(%367, %371) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %373 = "pphlo.shift_left"(%371, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %374 = "pphlo.shift_right_logical"(%371, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %375 = "pphlo.or"(%373, %374) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %376 = "pphlo.xor"(%372, %375) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %377 = "pphlo.add"(%372, %376) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %378 = "pphlo.add"(%377, %264) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %379 = "pphlo.shift_left"(%376, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %380 = "pphlo.shift_right_logical"(%376, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %381 = "pphlo.or"(%379, %380) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %382 = "pphlo.xor"(%377, %381) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %383 = "pphlo.add"(%382, %202) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %384 = "pphlo.concatenate"(%378, %383) {dimension = 0 : i64} : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<4x!pphlo.pub<ui32>>
    %385 = "pphlo.reshape"(%384) : (tensor<4x!pphlo.pub<ui32>>) -> tensor<2x2x!pphlo.pub<ui32>>
    %386 = "pphlo.slice"(%385) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pub<ui32>>) -> tensor<1x1x!pphlo.pub<ui32>>
    %387 = "pphlo.reshape"(%386) : (tensor<1x1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %388 = "pphlo.slice"(%385) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pub<ui32>>) -> tensor<1x2x!pphlo.pub<ui32>>
    %389 = "pphlo.reshape"(%388) : (tensor<1x2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %390 = "pphlo.slice"(%389) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %391 = "pphlo.add"(%390, %263) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %392 = "pphlo.add"(%387, %391) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %393 = "pphlo.shift_left"(%391, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %394 = "pphlo.shift_right_logical"(%391, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %395 = "pphlo.or"(%393, %394) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %396 = "pphlo.xor"(%392, %395) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %397 = "pphlo.add"(%392, %396) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %398 = "pphlo.shift_left"(%396, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %399 = "pphlo.shift_right_logical"(%396, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %400 = "pphlo.or"(%398, %399) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %401 = "pphlo.xor"(%397, %400) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %402 = "pphlo.add"(%397, %401) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %403 = "pphlo.shift_left"(%401, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %404 = "pphlo.shift_right_logical"(%401, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %405 = "pphlo.or"(%403, %404) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %406 = "pphlo.xor"(%402, %405) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %407 = "pphlo.add"(%402, %406) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %408 = "pphlo.add"(%407, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %409 = "pphlo.shift_left"(%406, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %410 = "pphlo.shift_right_logical"(%406, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %411 = "pphlo.or"(%409, %410) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %412 = "pphlo.xor"(%407, %411) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %413 = "pphlo.reshape"(%386) : (tensor<1x1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %414 = "pphlo.reshape"(%390) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %415 = "pphlo.xor"(%413, %414) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %416 = "pphlo.xor"(%415, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %417 = "pphlo.reshape"(%416) : (tensor<!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %418 = "pphlo.add"(%412, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %419 = "pphlo.add"(%418, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %420 = "pphlo.add"(%408, %419) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %421 = "pphlo.shift_left"(%419, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %422 = "pphlo.shift_right_logical"(%419, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %423 = "pphlo.or"(%421, %422) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %424 = "pphlo.xor"(%420, %423) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %425 = "pphlo.add"(%420, %424) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %426 = "pphlo.shift_left"(%424, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %427 = "pphlo.shift_right_logical"(%424, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %428 = "pphlo.or"(%426, %427) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %429 = "pphlo.xor"(%425, %428) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %430 = "pphlo.add"(%425, %429) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %431 = "pphlo.shift_left"(%429, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %432 = "pphlo.shift_right_logical"(%429, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %433 = "pphlo.or"(%431, %432) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %434 = "pphlo.xor"(%430, %433) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %435 = "pphlo.add"(%430, %434) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %436 = "pphlo.add"(%435, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %437 = "pphlo.shift_left"(%434, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %438 = "pphlo.shift_right_logical"(%434, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %439 = "pphlo.or"(%437, %438) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %440 = "pphlo.xor"(%435, %439) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %441 = "pphlo.add"(%440, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %442 = "pphlo.add"(%441, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %443 = "pphlo.add"(%436, %442) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %444 = "pphlo.shift_left"(%442, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %445 = "pphlo.shift_right_logical"(%442, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %446 = "pphlo.or"(%444, %445) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %447 = "pphlo.xor"(%443, %446) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %448 = "pphlo.add"(%443, %447) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %449 = "pphlo.shift_left"(%447, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %450 = "pphlo.shift_right_logical"(%447, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %451 = "pphlo.or"(%449, %450) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %452 = "pphlo.xor"(%448, %451) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %453 = "pphlo.add"(%448, %452) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %454 = "pphlo.shift_left"(%452, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %455 = "pphlo.shift_right_logical"(%452, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %456 = "pphlo.or"(%454, %455) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %457 = "pphlo.xor"(%453, %456) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %458 = "pphlo.add"(%453, %457) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %459 = "pphlo.add"(%458, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %460 = "pphlo.shift_left"(%457, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %461 = "pphlo.shift_right_logical"(%457, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %462 = "pphlo.or"(%460, %461) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %463 = "pphlo.xor"(%458, %462) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %464 = "pphlo.add"(%463, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %465 = "pphlo.add"(%464, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %466 = "pphlo.add"(%459, %465) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %467 = "pphlo.shift_left"(%465, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %468 = "pphlo.shift_right_logical"(%465, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %469 = "pphlo.or"(%467, %468) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %470 = "pphlo.xor"(%466, %469) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %471 = "pphlo.add"(%466, %470) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %472 = "pphlo.shift_left"(%470, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %473 = "pphlo.shift_right_logical"(%470, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %474 = "pphlo.or"(%472, %473) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %475 = "pphlo.xor"(%471, %474) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %476 = "pphlo.add"(%471, %475) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %477 = "pphlo.shift_left"(%475, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %478 = "pphlo.shift_right_logical"(%475, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %479 = "pphlo.or"(%477, %478) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %480 = "pphlo.xor"(%476, %479) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %481 = "pphlo.add"(%476, %480) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %482 = "pphlo.add"(%481, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %483 = "pphlo.shift_left"(%480, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %484 = "pphlo.shift_right_logical"(%480, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %485 = "pphlo.or"(%483, %484) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %486 = "pphlo.xor"(%481, %485) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %487 = "pphlo.add"(%486, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %488 = "pphlo.add"(%487, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %489 = "pphlo.add"(%482, %488) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %490 = "pphlo.shift_left"(%488, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %491 = "pphlo.shift_right_logical"(%488, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %492 = "pphlo.or"(%490, %491) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %493 = "pphlo.xor"(%489, %492) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %494 = "pphlo.add"(%489, %493) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %495 = "pphlo.shift_left"(%493, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %496 = "pphlo.shift_right_logical"(%493, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %497 = "pphlo.or"(%495, %496) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %498 = "pphlo.xor"(%494, %497) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %499 = "pphlo.add"(%494, %498) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %500 = "pphlo.shift_left"(%498, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %501 = "pphlo.shift_right_logical"(%498, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %502 = "pphlo.or"(%500, %501) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %503 = "pphlo.xor"(%499, %502) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %504 = "pphlo.add"(%499, %503) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %505 = "pphlo.add"(%504, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %506 = "pphlo.shift_left"(%503, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %507 = "pphlo.shift_right_logical"(%503, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %508 = "pphlo.or"(%506, %507) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %509 = "pphlo.xor"(%504, %508) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %510 = "pphlo.add"(%509, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %511 = "pphlo.add"(%510, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %512 = "pphlo.add"(%505, %511) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %513 = "pphlo.shift_left"(%511, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %514 = "pphlo.shift_right_logical"(%511, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %515 = "pphlo.or"(%513, %514) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %516 = "pphlo.xor"(%512, %515) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %517 = "pphlo.add"(%512, %516) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %518 = "pphlo.shift_left"(%516, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %519 = "pphlo.shift_right_logical"(%516, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %520 = "pphlo.or"(%518, %519) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %521 = "pphlo.xor"(%517, %520) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %522 = "pphlo.add"(%517, %521) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %523 = "pphlo.shift_left"(%521, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %524 = "pphlo.shift_right_logical"(%521, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %525 = "pphlo.or"(%523, %524) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %526 = "pphlo.xor"(%522, %525) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %527 = "pphlo.add"(%522, %526) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %528 = "pphlo.add"(%510, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %529 = "pphlo.add"(%527, %528) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %530 = "pphlo.shift_left"(%526, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %531 = "pphlo.shift_right_logical"(%526, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %532 = "pphlo.or"(%530, %531) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %533 = "pphlo.xor"(%527, %532) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %534 = "pphlo.reshape"(%505) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %535 = "pphlo.reshape"(%528) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %536 = "pphlo.xor"(%534, %535) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %537 = "pphlo.xor"(%536, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %538 = "pphlo.reshape"(%537) : (tensor<!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %539 = "pphlo.add"(%533, %538) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %540 = "pphlo.add"(%539, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %541 = "pphlo.add"(%529, %540) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %542 = "pphlo.shift_left"(%540, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %543 = "pphlo.shift_right_logical"(%540, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %544 = "pphlo.or"(%542, %543) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %545 = "pphlo.xor"(%541, %544) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %546 = "pphlo.add"(%541, %545) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %547 = "pphlo.shift_left"(%545, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %548 = "pphlo.shift_right_logical"(%545, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %549 = "pphlo.or"(%547, %548) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %550 = "pphlo.xor"(%546, %549) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %551 = "pphlo.add"(%546, %550) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %552 = "pphlo.shift_left"(%550, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %553 = "pphlo.shift_right_logical"(%550, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %554 = "pphlo.or"(%552, %553) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %555 = "pphlo.xor"(%551, %554) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %556 = "pphlo.add"(%551, %555) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %557 = "pphlo.add"(%556, %538) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %558 = "pphlo.shift_left"(%555, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %559 = "pphlo.shift_right_logical"(%555, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %560 = "pphlo.or"(%558, %559) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %561 = "pphlo.xor"(%556, %560) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %562 = "pphlo.add"(%561, %505) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %563 = "pphlo.add"(%562, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %564 = "pphlo.add"(%557, %563) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %565 = "pphlo.shift_left"(%563, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %566 = "pphlo.shift_right_logical"(%563, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %567 = "pphlo.or"(%565, %566) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %568 = "pphlo.xor"(%564, %567) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %569 = "pphlo.add"(%564, %568) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %570 = "pphlo.shift_left"(%568, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %571 = "pphlo.shift_right_logical"(%568, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %572 = "pphlo.or"(%570, %571) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %573 = "pphlo.xor"(%569, %572) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %574 = "pphlo.add"(%569, %573) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %575 = "pphlo.shift_left"(%573, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %576 = "pphlo.shift_right_logical"(%573, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %577 = "pphlo.or"(%575, %576) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %578 = "pphlo.xor"(%574, %577) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %579 = "pphlo.add"(%574, %578) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %580 = "pphlo.add"(%579, %505) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %581 = "pphlo.shift_left"(%578, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %582 = "pphlo.shift_right_logical"(%578, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %583 = "pphlo.or"(%581, %582) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %584 = "pphlo.xor"(%579, %583) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %585 = "pphlo.add"(%584, %528) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %586 = "pphlo.add"(%585, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %587 = "pphlo.add"(%580, %586) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %588 = "pphlo.shift_left"(%586, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %589 = "pphlo.shift_right_logical"(%586, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %590 = "pphlo.or"(%588, %589) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %591 = "pphlo.xor"(%587, %590) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %592 = "pphlo.add"(%587, %591) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %593 = "pphlo.shift_left"(%591, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %594 = "pphlo.shift_right_logical"(%591, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %595 = "pphlo.or"(%593, %594) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %596 = "pphlo.xor"(%592, %595) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %597 = "pphlo.add"(%592, %596) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %598 = "pphlo.shift_left"(%596, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %599 = "pphlo.shift_right_logical"(%596, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %600 = "pphlo.or"(%598, %599) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %601 = "pphlo.xor"(%597, %600) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %602 = "pphlo.add"(%597, %601) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %603 = "pphlo.add"(%602, %528) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %604 = "pphlo.shift_left"(%601, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %605 = "pphlo.shift_right_logical"(%601, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %606 = "pphlo.or"(%604, %605) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %607 = "pphlo.xor"(%602, %606) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %608 = "pphlo.add"(%607, %538) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %609 = "pphlo.add"(%608, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %610 = "pphlo.add"(%603, %609) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %611 = "pphlo.shift_left"(%609, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %612 = "pphlo.shift_right_logical"(%609, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %613 = "pphlo.or"(%611, %612) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %614 = "pphlo.xor"(%610, %613) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %615 = "pphlo.add"(%610, %614) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %616 = "pphlo.shift_left"(%614, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %617 = "pphlo.shift_right_logical"(%614, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %618 = "pphlo.or"(%616, %617) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %619 = "pphlo.xor"(%615, %618) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %620 = "pphlo.add"(%615, %619) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %621 = "pphlo.shift_left"(%619, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %622 = "pphlo.shift_right_logical"(%619, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %623 = "pphlo.or"(%621, %622) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %624 = "pphlo.xor"(%620, %623) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %625 = "pphlo.add"(%620, %624) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %626 = "pphlo.add"(%625, %538) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %627 = "pphlo.reshape"(%626) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %628 = "pphlo.broadcast"(%627) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %629 = "pphlo.add"(%273, %628) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %630 = "pphlo.slice"(%272) {limit_indices = dense<288> : tensor<1xi64>, start_indices = dense<144> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<288x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %631 = "pphlo.shift_left"(%624, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %632 = "pphlo.shift_right_logical"(%624, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %633 = "pphlo.or"(%631, %632) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %634 = "pphlo.xor"(%625, %633) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %635 = "pphlo.add"(%634, %505) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %636 = "pphlo.add"(%635, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %637 = "pphlo.reshape"(%636) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %638 = "pphlo.broadcast"(%637) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %639 = "pphlo.add"(%630, %638) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %640 = "pphlo.add"(%629, %639) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %641 = "pphlo.shift_left"(%639, %254) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %642 = "pphlo.shift_right_logical"(%639, %253) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %643 = "pphlo.or"(%641, %642) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %644 = "pphlo.xor"(%640, %643) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %645 = "pphlo.add"(%640, %644) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %646 = "pphlo.shift_left"(%644, %252) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %647 = "pphlo.shift_right_logical"(%644, %251) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %648 = "pphlo.or"(%646, %647) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %649 = "pphlo.xor"(%645, %648) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %650 = "pphlo.add"(%645, %649) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %651 = "pphlo.shift_left"(%649, %249) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %652 = "pphlo.shift_right_logical"(%649, %250) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %653 = "pphlo.or"(%651, %652) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %654 = "pphlo.xor"(%650, %653) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %655 = "pphlo.add"(%650, %654) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %656 = "pphlo.add"(%655, %638) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %657 = "pphlo.shift_left"(%654, %250) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %658 = "pphlo.shift_right_logical"(%654, %249) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %659 = "pphlo.or"(%657, %658) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %660 = "pphlo.xor"(%655, %659) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %661 = "pphlo.xor"(%627, %637) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %662 = "pphlo.xor"(%661, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %663 = "pphlo.broadcast"(%662) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %664 = "pphlo.add"(%660, %663) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %665 = "pphlo.add"(%664, %262) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %666 = "pphlo.add"(%656, %665) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %667 = "pphlo.shift_left"(%665, %251) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %668 = "pphlo.shift_right_logical"(%665, %252) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %669 = "pphlo.or"(%667, %668) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %670 = "pphlo.xor"(%666, %669) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %671 = "pphlo.add"(%666, %670) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %672 = "pphlo.shift_left"(%670, %260) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %673 = "pphlo.shift_right_logical"(%670, %259) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %674 = "pphlo.or"(%672, %673) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %675 = "pphlo.xor"(%671, %674) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %676 = "pphlo.add"(%671, %675) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %677 = "pphlo.shift_left"(%675, %258) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %678 = "pphlo.shift_right_logical"(%675, %258) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %679 = "pphlo.or"(%677, %678) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %680 = "pphlo.xor"(%676, %679) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %681 = "pphlo.add"(%676, %680) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %682 = "pphlo.add"(%681, %663) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %683 = "pphlo.shift_left"(%680, %257) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %684 = "pphlo.shift_right_logical"(%680, %256) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %685 = "pphlo.or"(%683, %684) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %686 = "pphlo.xor"(%681, %685) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %687 = "pphlo.add"(%686, %628) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %688 = "pphlo.add"(%687, %261) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %689 = "pphlo.add"(%682, %688) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %690 = "pphlo.shift_left"(%688, %254) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %691 = "pphlo.shift_right_logical"(%688, %253) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %692 = "pphlo.or"(%690, %691) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %693 = "pphlo.xor"(%689, %692) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %694 = "pphlo.add"(%689, %693) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %695 = "pphlo.shift_left"(%693, %252) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %696 = "pphlo.shift_right_logical"(%693, %251) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %697 = "pphlo.or"(%695, %696) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %698 = "pphlo.xor"(%694, %697) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %699 = "pphlo.add"(%694, %698) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %700 = "pphlo.shift_left"(%698, %249) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %701 = "pphlo.shift_right_logical"(%698, %250) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %702 = "pphlo.or"(%700, %701) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %703 = "pphlo.xor"(%699, %702) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %704 = "pphlo.add"(%699, %703) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %705 = "pphlo.add"(%704, %628) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %706 = "pphlo.shift_left"(%703, %250) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %707 = "pphlo.shift_right_logical"(%703, %249) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %708 = "pphlo.or"(%706, %707) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %709 = "pphlo.xor"(%704, %708) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %710 = "pphlo.add"(%709, %638) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %711 = "pphlo.add"(%710, %259) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %712 = "pphlo.add"(%705, %711) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %713 = "pphlo.shift_left"(%711, %251) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %714 = "pphlo.shift_right_logical"(%711, %252) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %715 = "pphlo.or"(%713, %714) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %716 = "pphlo.xor"(%712, %715) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %717 = "pphlo.add"(%712, %716) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %718 = "pphlo.shift_left"(%716, %260) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %719 = "pphlo.shift_right_logical"(%716, %259) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %720 = "pphlo.or"(%718, %719) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %721 = "pphlo.xor"(%717, %720) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %722 = "pphlo.add"(%717, %721) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %723 = "pphlo.shift_left"(%721, %258) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %724 = "pphlo.shift_right_logical"(%721, %258) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %725 = "pphlo.or"(%723, %724) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %726 = "pphlo.xor"(%722, %725) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %727 = "pphlo.add"(%722, %726) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %728 = "pphlo.add"(%727, %638) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %729 = "pphlo.shift_left"(%726, %257) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %730 = "pphlo.shift_right_logical"(%726, %256) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %731 = "pphlo.or"(%729, %730) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %732 = "pphlo.xor"(%727, %731) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %733 = "pphlo.add"(%732, %663) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %734 = "pphlo.add"(%733, %255) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %735 = "pphlo.add"(%728, %734) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %736 = "pphlo.shift_left"(%734, %254) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %737 = "pphlo.shift_right_logical"(%734, %253) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %738 = "pphlo.or"(%736, %737) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %739 = "pphlo.xor"(%735, %738) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %740 = "pphlo.add"(%735, %739) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %741 = "pphlo.shift_left"(%739, %252) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %742 = "pphlo.shift_right_logical"(%739, %251) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %743 = "pphlo.or"(%741, %742) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %744 = "pphlo.xor"(%740, %743) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %745 = "pphlo.add"(%740, %744) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %746 = "pphlo.shift_left"(%744, %249) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %747 = "pphlo.shift_right_logical"(%744, %250) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %748 = "pphlo.or"(%746, %747) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %749 = "pphlo.xor"(%745, %748) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %750 = "pphlo.add"(%745, %749) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %751 = "pphlo.add"(%750, %663) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %752 = "pphlo.shift_left"(%749, %250) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %753 = "pphlo.shift_right_logical"(%749, %249) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %754 = "pphlo.or"(%752, %753) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %755 = "pphlo.xor"(%750, %754) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %756 = "pphlo.add"(%755, %628) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %757 = "pphlo.add"(%756, %248) : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<144x!pphlo.pub<ui32>>
    %758 = "pphlo.concatenate"(%751, %757) {dimension = 0 : i64} : (tensor<144x!pphlo.pub<ui32>>, tensor<144x!pphlo.pub<ui32>>) -> tensor<288x!pphlo.pub<ui32>>
    %759 = "pphlo.shift_right_logical"(%758, %247) : (tensor<288x!pphlo.pub<ui32>>, tensor<288x!pphlo.pub<ui32>>) -> tensor<288x!pphlo.pub<ui32>>
    %760 = "pphlo.or"(%759, %246) : (tensor<288x!pphlo.pub<ui32>>, tensor<288x!pphlo.pub<ui32>>) -> tensor<288x!pphlo.pub<ui32>>
    %761 = "pphlo.bitcast_convert"(%760) {elsize = 32 : i64} : (tensor<288x!pphlo.pub<ui32>>) -> tensor<288x!pphlo.pub<f32>>
    %762 = "pphlo.add"(%761, %245) : (tensor<288x!pphlo.pub<f32>>, tensor<288x!pphlo.pub<f32>>) -> tensor<288x!pphlo.pub<f32>>
    %763 = "pphlo.multiply"(%762, %244) : (tensor<288x!pphlo.pub<f32>>, tensor<288x!pphlo.pub<f32>>) -> tensor<288x!pphlo.pub<f32>>
    %764 = "pphlo.add"(%763, %243) : (tensor<288x!pphlo.pub<f32>>, tensor<288x!pphlo.pub<f32>>) -> tensor<288x!pphlo.pub<f32>>
    %765 = "pphlo.maximum"(%764, %243) : (tensor<288x!pphlo.pub<f32>>, tensor<288x!pphlo.pub<f32>>) -> tensor<288x!pphlo.pub<f32>>
    %766 = "pphlo.reshape"(%765) : (tensor<288x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %767 = "pphlo.abs"(%766) : (tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %768 = "pphlo.equal"(%767, %242) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<i1>>
    %769 = "pphlo.multiply"(%766, %241) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %770 = "pphlo.negate"(%766) : (tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %771 = "pphlo.multiply"(%770, %766) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %772 = "pphlo.log_plus_one"(%771) : (tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %773 = "pphlo.negate"(%772) : (tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %774 = "pphlo.less"(%773, %240) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<i1>>
    %775 = "pphlo.select"(%774, %239, %238) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %776 = "pphlo.select"(%774, %237, %236) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %777 = "pphlo.select"(%774, %235, %234) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %778 = "pphlo.select"(%774, %233, %232) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %779 = "pphlo.select"(%774, %231, %230) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %780 = "pphlo.select"(%774, %229, %228) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %781 = "pphlo.select"(%774, %227, %226) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %782 = "pphlo.select"(%774, %225, %224) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %783 = "pphlo.select"(%774, %223, %222) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %784 = "pphlo.add"(%773, %221) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %785 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<3x3x1x32xf32>} : () -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %786 = "pphlo.power"(%773, %785) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %787 = "pphlo.add"(%786, %220) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %788 = "pphlo.select"(%774, %784, %787) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %789 = "pphlo.multiply"(%783, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %790 = "pphlo.add"(%782, %789) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %791 = "pphlo.multiply"(%790, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %792 = "pphlo.add"(%781, %791) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %793 = "pphlo.multiply"(%792, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %794 = "pphlo.add"(%780, %793) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %795 = "pphlo.multiply"(%794, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %796 = "pphlo.add"(%779, %795) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %797 = "pphlo.multiply"(%796, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %798 = "pphlo.add"(%778, %797) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %799 = "pphlo.multiply"(%798, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %800 = "pphlo.add"(%777, %799) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %801 = "pphlo.multiply"(%800, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %802 = "pphlo.add"(%776, %801) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %803 = "pphlo.multiply"(%802, %788) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %804 = "pphlo.add"(%775, %803) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %805 = "pphlo.multiply"(%804, %766) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %806 = "pphlo.select"(%768, %769, %805) : (tensor<3x3x1x32x!pphlo.pub<i1>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %807 = "pphlo.multiply"(%806, %219) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %808 = "pphlo.clamp"(%267, %807, %218) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %809 = "pphlo.multiply"(%808, %217) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %810 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<30x!pphlo.pub<ui32>>
    %811 = "pphlo.slice"(%810) {limit_indices = dense<15> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<30x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %812 = "pphlo.slice"(%378) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %813 = "pphlo.reshape"(%812) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %814 = "pphlo.broadcast"(%813) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %815 = "pphlo.add"(%275, %814) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %816 = "pphlo.slice"(%378) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %817 = "pphlo.reshape"(%816) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %818 = "pphlo.broadcast"(%817) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %819 = "pphlo.add"(%276, %818) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %820 = "pphlo.add"(%815, %819) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %821 = "pphlo.shift_left"(%819, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %822 = "pphlo.shift_right_logical"(%819, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %823 = "pphlo.or"(%821, %822) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %824 = "pphlo.xor"(%820, %823) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %825 = "pphlo.add"(%820, %824) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %826 = "pphlo.shift_left"(%824, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %827 = "pphlo.shift_right_logical"(%824, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %828 = "pphlo.or"(%826, %827) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %829 = "pphlo.xor"(%825, %828) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %830 = "pphlo.add"(%825, %829) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %831 = "pphlo.shift_left"(%829, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %832 = "pphlo.shift_right_logical"(%829, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %833 = "pphlo.or"(%831, %832) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %834 = "pphlo.xor"(%830, %833) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %835 = "pphlo.add"(%830, %834) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %836 = "pphlo.add"(%835, %818) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %837 = "pphlo.shift_left"(%834, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %838 = "pphlo.shift_right_logical"(%834, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %839 = "pphlo.or"(%837, %838) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %840 = "pphlo.xor"(%835, %839) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %841 = "pphlo.xor"(%813, %817) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %842 = "pphlo.xor"(%841, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %843 = "pphlo.broadcast"(%842) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %844 = "pphlo.add"(%840, %843) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %845 = "pphlo.add"(%844, %216) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %846 = "pphlo.add"(%836, %845) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %847 = "pphlo.shift_left"(%845, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %848 = "pphlo.shift_right_logical"(%845, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %849 = "pphlo.or"(%847, %848) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %850 = "pphlo.xor"(%846, %849) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %851 = "pphlo.add"(%846, %850) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %852 = "pphlo.shift_left"(%850, %214) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %853 = "pphlo.shift_right_logical"(%850, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %854 = "pphlo.or"(%852, %853) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %855 = "pphlo.xor"(%851, %854) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %856 = "pphlo.add"(%851, %855) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %857 = "pphlo.shift_left"(%855, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %858 = "pphlo.shift_right_logical"(%855, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %859 = "pphlo.or"(%857, %858) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %860 = "pphlo.xor"(%856, %859) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %861 = "pphlo.add"(%856, %860) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %862 = "pphlo.add"(%861, %843) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %863 = "pphlo.shift_left"(%860, %211) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %864 = "pphlo.shift_right_logical"(%860, %210) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %865 = "pphlo.or"(%863, %864) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %866 = "pphlo.xor"(%861, %865) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %867 = "pphlo.add"(%866, %814) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %868 = "pphlo.add"(%867, %215) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %869 = "pphlo.add"(%862, %868) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %870 = "pphlo.shift_left"(%868, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %871 = "pphlo.shift_right_logical"(%868, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %872 = "pphlo.or"(%870, %871) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %873 = "pphlo.xor"(%869, %872) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %874 = "pphlo.add"(%869, %873) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %875 = "pphlo.shift_left"(%873, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %876 = "pphlo.shift_right_logical"(%873, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %877 = "pphlo.or"(%875, %876) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %878 = "pphlo.xor"(%874, %877) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %879 = "pphlo.add"(%874, %878) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %880 = "pphlo.shift_left"(%878, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %881 = "pphlo.shift_right_logical"(%878, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %882 = "pphlo.or"(%880, %881) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %883 = "pphlo.xor"(%879, %882) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %884 = "pphlo.add"(%879, %883) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %885 = "pphlo.add"(%884, %814) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %886 = "pphlo.shift_left"(%883, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %887 = "pphlo.shift_right_logical"(%883, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %888 = "pphlo.or"(%886, %887) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %889 = "pphlo.xor"(%884, %888) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %890 = "pphlo.add"(%889, %818) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %891 = "pphlo.add"(%890, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %892 = "pphlo.add"(%885, %891) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %893 = "pphlo.shift_left"(%891, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %894 = "pphlo.shift_right_logical"(%891, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %895 = "pphlo.or"(%893, %894) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %896 = "pphlo.xor"(%892, %895) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %897 = "pphlo.add"(%892, %896) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %898 = "pphlo.shift_left"(%896, %214) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %899 = "pphlo.shift_right_logical"(%896, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %900 = "pphlo.or"(%898, %899) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %901 = "pphlo.xor"(%897, %900) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %902 = "pphlo.add"(%897, %901) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %903 = "pphlo.shift_left"(%901, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %904 = "pphlo.shift_right_logical"(%901, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %905 = "pphlo.or"(%903, %904) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %906 = "pphlo.xor"(%902, %905) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %907 = "pphlo.add"(%902, %906) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %908 = "pphlo.add"(%907, %818) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %909 = "pphlo.shift_left"(%906, %211) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %910 = "pphlo.shift_right_logical"(%906, %210) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %911 = "pphlo.or"(%909, %910) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %912 = "pphlo.xor"(%907, %911) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %913 = "pphlo.add"(%912, %843) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %914 = "pphlo.add"(%913, %209) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %915 = "pphlo.add"(%908, %914) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %916 = "pphlo.shift_left"(%914, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %917 = "pphlo.shift_right_logical"(%914, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %918 = "pphlo.or"(%916, %917) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %919 = "pphlo.xor"(%915, %918) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %920 = "pphlo.add"(%915, %919) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %921 = "pphlo.shift_left"(%919, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %922 = "pphlo.shift_right_logical"(%919, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %923 = "pphlo.or"(%921, %922) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %924 = "pphlo.xor"(%920, %923) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %925 = "pphlo.add"(%920, %924) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %926 = "pphlo.shift_left"(%924, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %927 = "pphlo.shift_right_logical"(%924, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %928 = "pphlo.or"(%926, %927) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %929 = "pphlo.xor"(%925, %928) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %930 = "pphlo.add"(%925, %929) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %931 = "pphlo.add"(%930, %843) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %932 = "pphlo.shift_left"(%929, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %933 = "pphlo.shift_right_logical"(%929, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %934 = "pphlo.or"(%932, %933) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %935 = "pphlo.xor"(%930, %934) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %936 = "pphlo.add"(%935, %814) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %937 = "pphlo.add"(%936, %202) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %938 = "pphlo.concatenate"(%931, %937) {dimension = 0 : i64} : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<4x!pphlo.pub<ui32>>
    %939 = "pphlo.reshape"(%938) : (tensor<4x!pphlo.pub<ui32>>) -> tensor<2x2x!pphlo.pub<ui32>>
    %940 = "pphlo.slice"(%939) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pub<ui32>>) -> tensor<1x1x!pphlo.pub<ui32>>
    %941 = "pphlo.reshape"(%940) : (tensor<1x1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %942 = "pphlo.broadcast"(%941) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %943 = "pphlo.add"(%275, %942) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %944 = "pphlo.slice"(%939) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pub<ui32>>) -> tensor<1x2x!pphlo.pub<ui32>>
    %945 = "pphlo.reshape"(%944) : (tensor<1x2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %946 = "pphlo.slice"(%945) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %947 = "pphlo.reshape"(%946) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %948 = "pphlo.broadcast"(%947) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %949 = "pphlo.add"(%276, %948) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %950 = "pphlo.add"(%943, %949) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %951 = "pphlo.shift_left"(%949, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %952 = "pphlo.shift_right_logical"(%949, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %953 = "pphlo.or"(%951, %952) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %954 = "pphlo.xor"(%950, %953) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %955 = "pphlo.add"(%950, %954) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %956 = "pphlo.shift_left"(%954, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %957 = "pphlo.shift_right_logical"(%954, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %958 = "pphlo.or"(%956, %957) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %959 = "pphlo.xor"(%955, %958) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %960 = "pphlo.add"(%955, %959) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %961 = "pphlo.shift_left"(%959, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %962 = "pphlo.shift_right_logical"(%959, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %963 = "pphlo.or"(%961, %962) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %964 = "pphlo.xor"(%960, %963) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %965 = "pphlo.add"(%960, %964) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %966 = "pphlo.add"(%965, %948) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %967 = "pphlo.shift_left"(%964, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %968 = "pphlo.shift_right_logical"(%964, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %969 = "pphlo.or"(%967, %968) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %970 = "pphlo.xor"(%965, %969) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %971 = "pphlo.xor"(%941, %947) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %972 = "pphlo.xor"(%971, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %973 = "pphlo.broadcast"(%972) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %974 = "pphlo.add"(%970, %973) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %975 = "pphlo.add"(%974, %216) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %976 = "pphlo.add"(%966, %975) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %977 = "pphlo.shift_left"(%975, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %978 = "pphlo.shift_right_logical"(%975, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %979 = "pphlo.or"(%977, %978) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %980 = "pphlo.xor"(%976, %979) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %981 = "pphlo.add"(%976, %980) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %982 = "pphlo.shift_left"(%980, %214) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %983 = "pphlo.shift_right_logical"(%980, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %984 = "pphlo.or"(%982, %983) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %985 = "pphlo.xor"(%981, %984) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %986 = "pphlo.add"(%981, %985) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %987 = "pphlo.shift_left"(%985, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %988 = "pphlo.shift_right_logical"(%985, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %989 = "pphlo.or"(%987, %988) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %990 = "pphlo.xor"(%986, %989) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %991 = "pphlo.add"(%986, %990) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %992 = "pphlo.add"(%991, %973) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %993 = "pphlo.shift_left"(%990, %211) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %994 = "pphlo.shift_right_logical"(%990, %210) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %995 = "pphlo.or"(%993, %994) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %996 = "pphlo.xor"(%991, %995) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %997 = "pphlo.add"(%996, %942) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %998 = "pphlo.add"(%997, %215) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %999 = "pphlo.add"(%992, %998) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1000 = "pphlo.shift_left"(%998, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1001 = "pphlo.shift_right_logical"(%998, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1002 = "pphlo.or"(%1000, %1001) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1003 = "pphlo.xor"(%999, %1002) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1004 = "pphlo.add"(%999, %1003) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1005 = "pphlo.shift_left"(%1003, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1006 = "pphlo.shift_right_logical"(%1003, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1007 = "pphlo.or"(%1005, %1006) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1008 = "pphlo.xor"(%1004, %1007) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1009 = "pphlo.add"(%1004, %1008) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1010 = "pphlo.shift_left"(%1008, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1011 = "pphlo.shift_right_logical"(%1008, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1012 = "pphlo.or"(%1010, %1011) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1013 = "pphlo.xor"(%1009, %1012) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1014 = "pphlo.add"(%1009, %1013) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1015 = "pphlo.add"(%1014, %942) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1016 = "pphlo.shift_left"(%1013, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1017 = "pphlo.shift_right_logical"(%1013, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1018 = "pphlo.or"(%1016, %1017) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1019 = "pphlo.xor"(%1014, %1018) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1020 = "pphlo.add"(%1019, %948) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1021 = "pphlo.add"(%1020, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1022 = "pphlo.add"(%1015, %1021) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1023 = "pphlo.shift_left"(%1021, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1024 = "pphlo.shift_right_logical"(%1021, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1025 = "pphlo.or"(%1023, %1024) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1026 = "pphlo.xor"(%1022, %1025) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1027 = "pphlo.add"(%1022, %1026) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1028 = "pphlo.shift_left"(%1026, %214) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1029 = "pphlo.shift_right_logical"(%1026, %213) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1030 = "pphlo.or"(%1028, %1029) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1031 = "pphlo.xor"(%1027, %1030) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1032 = "pphlo.add"(%1027, %1031) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1033 = "pphlo.shift_left"(%1031, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1034 = "pphlo.shift_right_logical"(%1031, %212) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1035 = "pphlo.or"(%1033, %1034) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1036 = "pphlo.xor"(%1032, %1035) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1037 = "pphlo.add"(%1032, %1036) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1038 = "pphlo.add"(%1037, %948) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1039 = "pphlo.shift_left"(%1036, %211) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1040 = "pphlo.shift_right_logical"(%1036, %210) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1041 = "pphlo.or"(%1039, %1040) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1042 = "pphlo.xor"(%1037, %1041) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1043 = "pphlo.add"(%1042, %973) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1044 = "pphlo.add"(%1043, %209) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1045 = "pphlo.add"(%1038, %1044) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1046 = "pphlo.shift_left"(%1044, %208) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1047 = "pphlo.shift_right_logical"(%1044, %207) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1048 = "pphlo.or"(%1046, %1047) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1049 = "pphlo.xor"(%1045, %1048) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1050 = "pphlo.add"(%1045, %1049) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1051 = "pphlo.shift_left"(%1049, %206) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1052 = "pphlo.shift_right_logical"(%1049, %205) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1053 = "pphlo.or"(%1051, %1052) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1054 = "pphlo.xor"(%1050, %1053) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1055 = "pphlo.add"(%1050, %1054) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1056 = "pphlo.shift_left"(%1054, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1057 = "pphlo.shift_right_logical"(%1054, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1058 = "pphlo.or"(%1056, %1057) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1059 = "pphlo.xor"(%1055, %1058) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1060 = "pphlo.add"(%1055, %1059) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1061 = "pphlo.add"(%1060, %973) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1062 = "pphlo.shift_left"(%1059, %204) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1063 = "pphlo.shift_right_logical"(%1059, %203) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1064 = "pphlo.or"(%1062, %1063) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1065 = "pphlo.xor"(%1060, %1064) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1066 = "pphlo.add"(%1065, %942) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1067 = "pphlo.add"(%1066, %202) : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1068 = "pphlo.concatenate"(%1061, %1067) {dimension = 0 : i64} : (tensor<2x!pphlo.pub<ui32>>, tensor<2x!pphlo.pub<ui32>>) -> tensor<4x!pphlo.pub<ui32>>
    %1069 = "pphlo.reshape"(%1068) : (tensor<4x!pphlo.pub<ui32>>) -> tensor<2x2x!pphlo.pub<ui32>>
    %1070 = "pphlo.slice"(%1069) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pub<ui32>>) -> tensor<1x1x!pphlo.pub<ui32>>
    %1071 = "pphlo.reshape"(%1070) : (tensor<1x1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1072 = "pphlo.broadcast"(%1071) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1073 = "pphlo.add"(%811, %1072) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1074 = "pphlo.slice"(%810) {limit_indices = dense<30> : tensor<1xi64>, start_indices = dense<15> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<30x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1075 = "pphlo.slice"(%1069) {limit_indices = dense<2> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2x!pphlo.pub<ui32>>) -> tensor<1x2x!pphlo.pub<ui32>>
    %1076 = "pphlo.reshape"(%1075) : (tensor<1x2x!pphlo.pub<ui32>>) -> tensor<2x!pphlo.pub<ui32>>
    %1077 = "pphlo.slice"(%1076) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1078 = "pphlo.reshape"(%1077) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1079 = "pphlo.broadcast"(%1078) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1080 = "pphlo.add"(%1074, %1079) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1081 = "pphlo.add"(%1073, %1080) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1082 = "pphlo.shift_left"(%1080, %193) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1083 = "pphlo.shift_right_logical"(%1080, %192) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1084 = "pphlo.or"(%1082, %1083) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1085 = "pphlo.xor"(%1081, %1084) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1086 = "pphlo.add"(%1081, %1085) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1087 = "pphlo.shift_left"(%1085, %191) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1088 = "pphlo.shift_right_logical"(%1085, %190) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1089 = "pphlo.or"(%1087, %1088) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1090 = "pphlo.xor"(%1086, %1089) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1091 = "pphlo.add"(%1086, %1090) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1092 = "pphlo.shift_left"(%1090, %188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1093 = "pphlo.shift_right_logical"(%1090, %189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1094 = "pphlo.or"(%1092, %1093) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1095 = "pphlo.xor"(%1091, %1094) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1096 = "pphlo.add"(%1091, %1095) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1097 = "pphlo.add"(%1096, %1079) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1098 = "pphlo.shift_left"(%1095, %189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1099 = "pphlo.shift_right_logical"(%1095, %188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1100 = "pphlo.or"(%1098, %1099) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1101 = "pphlo.xor"(%1096, %1100) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1102 = "pphlo.xor"(%1071, %1078) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1103 = "pphlo.xor"(%1102, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1104 = "pphlo.broadcast"(%1103) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1105 = "pphlo.add"(%1101, %1104) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1106 = "pphlo.add"(%1105, %201) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1107 = "pphlo.add"(%1097, %1106) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1108 = "pphlo.shift_left"(%1106, %190) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1109 = "pphlo.shift_right_logical"(%1106, %191) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1110 = "pphlo.or"(%1108, %1109) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1111 = "pphlo.xor"(%1107, %1110) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1112 = "pphlo.add"(%1107, %1111) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1113 = "pphlo.shift_left"(%1111, %199) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1114 = "pphlo.shift_right_logical"(%1111, %198) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1115 = "pphlo.or"(%1113, %1114) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1116 = "pphlo.xor"(%1112, %1115) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1117 = "pphlo.add"(%1112, %1116) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1118 = "pphlo.shift_left"(%1116, %197) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1119 = "pphlo.shift_right_logical"(%1116, %197) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1120 = "pphlo.or"(%1118, %1119) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1121 = "pphlo.xor"(%1117, %1120) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1122 = "pphlo.add"(%1117, %1121) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1123 = "pphlo.add"(%1122, %1104) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1124 = "pphlo.shift_left"(%1121, %196) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1125 = "pphlo.shift_right_logical"(%1121, %195) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1126 = "pphlo.or"(%1124, %1125) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1127 = "pphlo.xor"(%1122, %1126) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1128 = "pphlo.add"(%1127, %1072) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1129 = "pphlo.add"(%1128, %200) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1130 = "pphlo.add"(%1123, %1129) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1131 = "pphlo.shift_left"(%1129, %193) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1132 = "pphlo.shift_right_logical"(%1129, %192) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1133 = "pphlo.or"(%1131, %1132) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1134 = "pphlo.xor"(%1130, %1133) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1135 = "pphlo.add"(%1130, %1134) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1136 = "pphlo.shift_left"(%1134, %191) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1137 = "pphlo.shift_right_logical"(%1134, %190) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1138 = "pphlo.or"(%1136, %1137) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1139 = "pphlo.xor"(%1135, %1138) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1140 = "pphlo.add"(%1135, %1139) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1141 = "pphlo.shift_left"(%1139, %188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1142 = "pphlo.shift_right_logical"(%1139, %189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1143 = "pphlo.or"(%1141, %1142) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1144 = "pphlo.xor"(%1140, %1143) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1145 = "pphlo.add"(%1140, %1144) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1146 = "pphlo.add"(%1145, %1072) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1147 = "pphlo.shift_left"(%1144, %189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1148 = "pphlo.shift_right_logical"(%1144, %188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1149 = "pphlo.or"(%1147, %1148) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1150 = "pphlo.xor"(%1145, %1149) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1151 = "pphlo.add"(%1150, %1079) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1152 = "pphlo.add"(%1151, %198) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1153 = "pphlo.add"(%1146, %1152) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1154 = "pphlo.shift_left"(%1152, %190) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1155 = "pphlo.shift_right_logical"(%1152, %191) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1156 = "pphlo.or"(%1154, %1155) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1157 = "pphlo.xor"(%1153, %1156) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1158 = "pphlo.add"(%1153, %1157) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1159 = "pphlo.shift_left"(%1157, %199) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1160 = "pphlo.shift_right_logical"(%1157, %198) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1161 = "pphlo.or"(%1159, %1160) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1162 = "pphlo.xor"(%1158, %1161) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1163 = "pphlo.add"(%1158, %1162) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1164 = "pphlo.shift_left"(%1162, %197) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1165 = "pphlo.shift_right_logical"(%1162, %197) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1166 = "pphlo.or"(%1164, %1165) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1167 = "pphlo.xor"(%1163, %1166) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1168 = "pphlo.add"(%1163, %1167) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1169 = "pphlo.add"(%1168, %1079) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1170 = "pphlo.shift_left"(%1167, %196) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1171 = "pphlo.shift_right_logical"(%1167, %195) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1172 = "pphlo.or"(%1170, %1171) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1173 = "pphlo.xor"(%1168, %1172) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1174 = "pphlo.add"(%1173, %1104) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1175 = "pphlo.add"(%1174, %194) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1176 = "pphlo.add"(%1169, %1175) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1177 = "pphlo.shift_left"(%1175, %193) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1178 = "pphlo.shift_right_logical"(%1175, %192) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1179 = "pphlo.or"(%1177, %1178) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1180 = "pphlo.xor"(%1176, %1179) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1181 = "pphlo.add"(%1176, %1180) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1182 = "pphlo.shift_left"(%1180, %191) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1183 = "pphlo.shift_right_logical"(%1180, %190) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1184 = "pphlo.or"(%1182, %1183) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1185 = "pphlo.xor"(%1181, %1184) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1186 = "pphlo.add"(%1181, %1185) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1187 = "pphlo.shift_left"(%1185, %188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1188 = "pphlo.shift_right_logical"(%1185, %189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1189 = "pphlo.or"(%1187, %1188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1190 = "pphlo.xor"(%1186, %1189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1191 = "pphlo.add"(%1186, %1190) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1192 = "pphlo.add"(%1191, %1104) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1193 = "pphlo.shift_left"(%1190, %189) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1194 = "pphlo.shift_right_logical"(%1190, %188) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1195 = "pphlo.or"(%1193, %1194) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1196 = "pphlo.xor"(%1191, %1195) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1197 = "pphlo.add"(%1196, %1072) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1198 = "pphlo.add"(%1197, %187) : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<15x!pphlo.pub<ui32>>
    %1199 = "pphlo.concatenate"(%1192, %1198) {dimension = 0 : i64} : (tensor<15x!pphlo.pub<ui32>>, tensor<15x!pphlo.pub<ui32>>) -> tensor<30x!pphlo.pub<ui32>>
    %1200 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<30x!pphlo.pub<i32>>
    %1201:2 = "pphlo.sort"(%1199, %1200) ({
    ^bb0(%arg4: tensor<!pphlo.pub<ui32>>, %arg5: tensor<!pphlo.pub<ui32>>, %arg6: tensor<!pphlo.pub<i32>>, %arg7: tensor<!pphlo.pub<i32>>):
      %2555 = "pphlo.less"(%arg4, %arg5) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<i1>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<i1>>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<30x!pphlo.pub<ui32>>, tensor<30x!pphlo.pub<i32>>) -> (tensor<30x!pphlo.pub<ui32>>, tensor<30x!pphlo.pub<i32>>)
    %1202 = "pphlo.less"(%1201#1, %89) : (tensor<30x!pphlo.pub<i32>>, tensor<30x!pphlo.pub<i32>>) -> tensor<30x!pphlo.pub<i1>>
    %1203 = "pphlo.add"(%1201#1, %88) : (tensor<30x!pphlo.pub<i32>>, tensor<30x!pphlo.pub<i32>>) -> tensor<30x!pphlo.pub<i32>>
    %1204 = "pphlo.select"(%1202, %1203, %1201#1) : (tensor<30x!pphlo.pub<i1>>, tensor<30x!pphlo.pub<i32>>, tensor<30x!pphlo.pub<i32>>) -> tensor<30x!pphlo.pub<i32>>
    %1205 = "pphlo.reshape"(%1204) : (tensor<30x!pphlo.pub<i32>>) -> tensor<30x1x!pphlo.pub<i32>>
    %1206 = "pphlo.gather"(%arg0, %1205) {dimension_numbers = #pphlo.gather<offset_dims = [1, 2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 28, 28, 1]> : tensor<4xi64>} : (tensor<30x28x28x1x!pphlo.pub<f32>>, tensor<30x1x!pphlo.pub<i32>>) -> tensor<30x28x28x1x!pphlo.pub<f32>>
    %1207 = pphlo.convolution(%1206, %809) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x28x28x1x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<30x28x28x32x!pphlo.pub<f32>>
    %1208 = "pphlo.greater"(%1207, %17) : (tensor<30x28x28x32x!pphlo.pub<f32>>, tensor<30x28x28x32x!pphlo.pub<f32>>) -> tensor<30x28x28x32x!pphlo.pub<i1>>
    %1209 = "pphlo.maximum"(%1207, %17) : (tensor<30x28x28x32x!pphlo.pub<f32>>, tensor<30x28x28x32x!pphlo.pub<f32>>) -> tensor<30x28x28x32x!pphlo.pub<f32>>
    %1210 = "pphlo.reduce_window"(%1209, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<30x28x28x32x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x14x14x32x!pphlo.pub<f32>>
    %1211 = "pphlo.multiply"(%1210, %18) : (tensor<30x14x14x32x!pphlo.pub<f32>>, tensor<30x14x14x32x!pphlo.pub<f32>>) -> tensor<30x14x14x32x!pphlo.pub<f32>>
    %1212 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<18432x!pphlo.pub<ui32>>
    %1213 = "pphlo.slice"(%1212) {limit_indices = dense<9216> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<18432x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1214 = "pphlo.add"(%390, %185) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1215 = "pphlo.add"(%387, %1214) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1216 = "pphlo.shift_left"(%1214, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1217 = "pphlo.shift_right_logical"(%1214, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1218 = "pphlo.or"(%1216, %1217) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1219 = "pphlo.xor"(%1215, %1218) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1220 = "pphlo.add"(%1215, %1219) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1221 = "pphlo.shift_left"(%1219, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1222 = "pphlo.shift_right_logical"(%1219, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1223 = "pphlo.or"(%1221, %1222) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1224 = "pphlo.xor"(%1220, %1223) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1225 = "pphlo.add"(%1220, %1224) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1226 = "pphlo.shift_left"(%1224, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1227 = "pphlo.shift_right_logical"(%1224, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1228 = "pphlo.or"(%1226, %1227) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1229 = "pphlo.xor"(%1225, %1228) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1230 = "pphlo.add"(%1225, %1229) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1231 = "pphlo.add"(%1230, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1232 = "pphlo.shift_left"(%1229, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1233 = "pphlo.shift_right_logical"(%1229, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1234 = "pphlo.or"(%1232, %1233) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1235 = "pphlo.xor"(%1230, %1234) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1236 = "pphlo.add"(%1235, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1237 = "pphlo.add"(%1236, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1238 = "pphlo.add"(%1231, %1237) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1239 = "pphlo.shift_left"(%1237, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1240 = "pphlo.shift_right_logical"(%1237, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1241 = "pphlo.or"(%1239, %1240) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1242 = "pphlo.xor"(%1238, %1241) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1243 = "pphlo.add"(%1238, %1242) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1244 = "pphlo.shift_left"(%1242, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1245 = "pphlo.shift_right_logical"(%1242, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1246 = "pphlo.or"(%1244, %1245) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1247 = "pphlo.xor"(%1243, %1246) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1248 = "pphlo.add"(%1243, %1247) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1249 = "pphlo.shift_left"(%1247, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1250 = "pphlo.shift_right_logical"(%1247, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1251 = "pphlo.or"(%1249, %1250) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1252 = "pphlo.xor"(%1248, %1251) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1253 = "pphlo.add"(%1248, %1252) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1254 = "pphlo.add"(%1253, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1255 = "pphlo.shift_left"(%1252, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1256 = "pphlo.shift_right_logical"(%1252, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1257 = "pphlo.or"(%1255, %1256) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1258 = "pphlo.xor"(%1253, %1257) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1259 = "pphlo.add"(%1258, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1260 = "pphlo.add"(%1259, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1261 = "pphlo.add"(%1254, %1260) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1262 = "pphlo.shift_left"(%1260, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1263 = "pphlo.shift_right_logical"(%1260, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1264 = "pphlo.or"(%1262, %1263) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1265 = "pphlo.xor"(%1261, %1264) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1266 = "pphlo.add"(%1261, %1265) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1267 = "pphlo.shift_left"(%1265, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1268 = "pphlo.shift_right_logical"(%1265, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1269 = "pphlo.or"(%1267, %1268) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1270 = "pphlo.xor"(%1266, %1269) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1271 = "pphlo.add"(%1266, %1270) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1272 = "pphlo.shift_left"(%1270, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1273 = "pphlo.shift_right_logical"(%1270, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1274 = "pphlo.or"(%1272, %1273) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1275 = "pphlo.xor"(%1271, %1274) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1276 = "pphlo.add"(%1271, %1275) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1277 = "pphlo.add"(%1276, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1278 = "pphlo.shift_left"(%1275, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1279 = "pphlo.shift_right_logical"(%1275, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1280 = "pphlo.or"(%1278, %1279) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1281 = "pphlo.xor"(%1276, %1280) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1282 = "pphlo.add"(%1281, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1283 = "pphlo.add"(%1282, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1284 = "pphlo.add"(%1277, %1283) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1285 = "pphlo.shift_left"(%1283, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1286 = "pphlo.shift_right_logical"(%1283, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1287 = "pphlo.or"(%1285, %1286) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1288 = "pphlo.xor"(%1284, %1287) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1289 = "pphlo.add"(%1284, %1288) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1290 = "pphlo.shift_left"(%1288, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1291 = "pphlo.shift_right_logical"(%1288, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1292 = "pphlo.or"(%1290, %1291) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1293 = "pphlo.xor"(%1289, %1292) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1294 = "pphlo.add"(%1289, %1293) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1295 = "pphlo.shift_left"(%1293, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1296 = "pphlo.shift_right_logical"(%1293, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1297 = "pphlo.or"(%1295, %1296) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1298 = "pphlo.xor"(%1294, %1297) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1299 = "pphlo.add"(%1294, %1298) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1300 = "pphlo.add"(%1299, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1301 = "pphlo.shift_left"(%1298, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1302 = "pphlo.shift_right_logical"(%1298, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1303 = "pphlo.or"(%1301, %1302) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1304 = "pphlo.xor"(%1299, %1303) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1305 = "pphlo.add"(%1304, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1306 = "pphlo.add"(%1305, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1307 = "pphlo.add"(%1300, %1306) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1308 = "pphlo.shift_left"(%1306, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1309 = "pphlo.shift_right_logical"(%1306, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1310 = "pphlo.or"(%1308, %1309) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1311 = "pphlo.xor"(%1307, %1310) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1312 = "pphlo.add"(%1307, %1311) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1313 = "pphlo.shift_left"(%1311, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1314 = "pphlo.shift_right_logical"(%1311, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1315 = "pphlo.or"(%1313, %1314) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1316 = "pphlo.xor"(%1312, %1315) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1317 = "pphlo.add"(%1312, %1316) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1318 = "pphlo.shift_left"(%1316, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1319 = "pphlo.shift_right_logical"(%1316, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1320 = "pphlo.or"(%1318, %1319) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1321 = "pphlo.xor"(%1317, %1320) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1322 = "pphlo.add"(%1317, %1321) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1323 = "pphlo.add"(%1322, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1324 = "pphlo.shift_left"(%1321, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1325 = "pphlo.shift_right_logical"(%1321, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1326 = "pphlo.or"(%1324, %1325) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1327 = "pphlo.xor"(%1322, %1326) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1328 = "pphlo.add"(%1327, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1329 = "pphlo.add"(%1328, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1330 = "pphlo.add"(%1323, %1329) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1331 = "pphlo.shift_left"(%1329, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1332 = "pphlo.shift_right_logical"(%1329, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1333 = "pphlo.or"(%1331, %1332) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1334 = "pphlo.xor"(%1330, %1333) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1335 = "pphlo.add"(%1330, %1334) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1336 = "pphlo.shift_left"(%1334, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1337 = "pphlo.shift_right_logical"(%1334, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1338 = "pphlo.or"(%1336, %1337) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1339 = "pphlo.xor"(%1335, %1338) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1340 = "pphlo.add"(%1335, %1339) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1341 = "pphlo.shift_left"(%1339, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1342 = "pphlo.shift_right_logical"(%1339, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1343 = "pphlo.or"(%1341, %1342) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1344 = "pphlo.xor"(%1340, %1343) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1345 = "pphlo.add"(%1340, %1344) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1346 = "pphlo.add"(%1328, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1347 = "pphlo.add"(%1345, %1346) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1348 = "pphlo.shift_left"(%1344, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1349 = "pphlo.shift_right_logical"(%1344, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1350 = "pphlo.or"(%1348, %1349) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1351 = "pphlo.xor"(%1345, %1350) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1352 = "pphlo.reshape"(%1323) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1353 = "pphlo.reshape"(%1346) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1354 = "pphlo.xor"(%1352, %1353) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1355 = "pphlo.xor"(%1354, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1356 = "pphlo.reshape"(%1355) : (tensor<!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1357 = "pphlo.add"(%1351, %1356) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1358 = "pphlo.add"(%1357, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1359 = "pphlo.add"(%1347, %1358) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1360 = "pphlo.shift_left"(%1358, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1361 = "pphlo.shift_right_logical"(%1358, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1362 = "pphlo.or"(%1360, %1361) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1363 = "pphlo.xor"(%1359, %1362) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1364 = "pphlo.add"(%1359, %1363) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1365 = "pphlo.shift_left"(%1363, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1366 = "pphlo.shift_right_logical"(%1363, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1367 = "pphlo.or"(%1365, %1366) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1368 = "pphlo.xor"(%1364, %1367) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1369 = "pphlo.add"(%1364, %1368) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1370 = "pphlo.shift_left"(%1368, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1371 = "pphlo.shift_right_logical"(%1368, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1372 = "pphlo.or"(%1370, %1371) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1373 = "pphlo.xor"(%1369, %1372) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1374 = "pphlo.add"(%1369, %1373) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1375 = "pphlo.add"(%1374, %1356) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1376 = "pphlo.shift_left"(%1373, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1377 = "pphlo.shift_right_logical"(%1373, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1378 = "pphlo.or"(%1376, %1377) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1379 = "pphlo.xor"(%1374, %1378) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1380 = "pphlo.add"(%1379, %1323) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1381 = "pphlo.add"(%1380, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1382 = "pphlo.add"(%1375, %1381) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1383 = "pphlo.shift_left"(%1381, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1384 = "pphlo.shift_right_logical"(%1381, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1385 = "pphlo.or"(%1383, %1384) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1386 = "pphlo.xor"(%1382, %1385) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1387 = "pphlo.add"(%1382, %1386) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1388 = "pphlo.shift_left"(%1386, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1389 = "pphlo.shift_right_logical"(%1386, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1390 = "pphlo.or"(%1388, %1389) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1391 = "pphlo.xor"(%1387, %1390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1392 = "pphlo.add"(%1387, %1391) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1393 = "pphlo.shift_left"(%1391, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1394 = "pphlo.shift_right_logical"(%1391, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1395 = "pphlo.or"(%1393, %1394) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1396 = "pphlo.xor"(%1392, %1395) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1397 = "pphlo.add"(%1392, %1396) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1398 = "pphlo.add"(%1397, %1323) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1399 = "pphlo.shift_left"(%1396, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1400 = "pphlo.shift_right_logical"(%1396, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1401 = "pphlo.or"(%1399, %1400) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1402 = "pphlo.xor"(%1397, %1401) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1403 = "pphlo.add"(%1402, %1346) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1404 = "pphlo.add"(%1403, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1405 = "pphlo.add"(%1398, %1404) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1406 = "pphlo.shift_left"(%1404, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1407 = "pphlo.shift_right_logical"(%1404, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1408 = "pphlo.or"(%1406, %1407) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1409 = "pphlo.xor"(%1405, %1408) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1410 = "pphlo.add"(%1405, %1409) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1411 = "pphlo.shift_left"(%1409, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1412 = "pphlo.shift_right_logical"(%1409, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1413 = "pphlo.or"(%1411, %1412) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1414 = "pphlo.xor"(%1410, %1413) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1415 = "pphlo.add"(%1410, %1414) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1416 = "pphlo.shift_left"(%1414, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1417 = "pphlo.shift_right_logical"(%1414, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1418 = "pphlo.or"(%1416, %1417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1419 = "pphlo.xor"(%1415, %1418) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1420 = "pphlo.add"(%1415, %1419) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1421 = "pphlo.add"(%1420, %1346) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1422 = "pphlo.shift_left"(%1419, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1423 = "pphlo.shift_right_logical"(%1419, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1424 = "pphlo.or"(%1422, %1423) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1425 = "pphlo.xor"(%1420, %1424) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1426 = "pphlo.add"(%1425, %1356) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1427 = "pphlo.add"(%1426, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1428 = "pphlo.add"(%1421, %1427) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1429 = "pphlo.shift_left"(%1427, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1430 = "pphlo.shift_right_logical"(%1427, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1431 = "pphlo.or"(%1429, %1430) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1432 = "pphlo.xor"(%1428, %1431) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1433 = "pphlo.add"(%1428, %1432) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1434 = "pphlo.shift_left"(%1432, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1435 = "pphlo.shift_right_logical"(%1432, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1436 = "pphlo.or"(%1434, %1435) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1437 = "pphlo.xor"(%1433, %1436) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1438 = "pphlo.add"(%1433, %1437) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1439 = "pphlo.shift_left"(%1437, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1440 = "pphlo.shift_right_logical"(%1437, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1441 = "pphlo.or"(%1439, %1440) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1442 = "pphlo.xor"(%1438, %1441) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1443 = "pphlo.add"(%1438, %1442) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1444 = "pphlo.add"(%1443, %1356) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1445 = "pphlo.reshape"(%1444) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1446 = "pphlo.broadcast"(%1445) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1447 = "pphlo.add"(%1213, %1446) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1448 = "pphlo.slice"(%1212) {limit_indices = dense<18432> : tensor<1xi64>, start_indices = dense<9216> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<18432x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1449 = "pphlo.shift_left"(%1442, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1450 = "pphlo.shift_right_logical"(%1442, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1451 = "pphlo.or"(%1449, %1450) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1452 = "pphlo.xor"(%1443, %1451) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1453 = "pphlo.add"(%1452, %1323) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1454 = "pphlo.add"(%1453, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1455 = "pphlo.reshape"(%1454) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1456 = "pphlo.broadcast"(%1455) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1457 = "pphlo.add"(%1448, %1456) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1458 = "pphlo.add"(%1447, %1457) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1459 = "pphlo.shift_left"(%1457, %176) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1460 = "pphlo.shift_right_logical"(%1457, %175) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1461 = "pphlo.or"(%1459, %1460) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1462 = "pphlo.xor"(%1458, %1461) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1463 = "pphlo.add"(%1458, %1462) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1464 = "pphlo.shift_left"(%1462, %174) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1465 = "pphlo.shift_right_logical"(%1462, %173) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1466 = "pphlo.or"(%1464, %1465) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1467 = "pphlo.xor"(%1463, %1466) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1468 = "pphlo.add"(%1463, %1467) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1469 = "pphlo.shift_left"(%1467, %171) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1470 = "pphlo.shift_right_logical"(%1467, %172) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1471 = "pphlo.or"(%1469, %1470) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1472 = "pphlo.xor"(%1468, %1471) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1473 = "pphlo.add"(%1468, %1472) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1474 = "pphlo.add"(%1473, %1456) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1475 = "pphlo.shift_left"(%1472, %172) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1476 = "pphlo.shift_right_logical"(%1472, %171) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1477 = "pphlo.or"(%1475, %1476) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1478 = "pphlo.xor"(%1473, %1477) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1479 = "pphlo.xor"(%1445, %1455) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1480 = "pphlo.xor"(%1479, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1481 = "pphlo.broadcast"(%1480) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1482 = "pphlo.add"(%1478, %1481) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1483 = "pphlo.add"(%1482, %184) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1484 = "pphlo.add"(%1474, %1483) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1485 = "pphlo.shift_left"(%1483, %173) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1486 = "pphlo.shift_right_logical"(%1483, %174) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1487 = "pphlo.or"(%1485, %1486) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1488 = "pphlo.xor"(%1484, %1487) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1489 = "pphlo.add"(%1484, %1488) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1490 = "pphlo.shift_left"(%1488, %182) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1491 = "pphlo.shift_right_logical"(%1488, %181) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1492 = "pphlo.or"(%1490, %1491) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1493 = "pphlo.xor"(%1489, %1492) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1494 = "pphlo.add"(%1489, %1493) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1495 = "pphlo.shift_left"(%1493, %180) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1496 = "pphlo.shift_right_logical"(%1493, %180) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1497 = "pphlo.or"(%1495, %1496) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1498 = "pphlo.xor"(%1494, %1497) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1499 = "pphlo.add"(%1494, %1498) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1500 = "pphlo.add"(%1499, %1481) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1501 = "pphlo.shift_left"(%1498, %179) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1502 = "pphlo.shift_right_logical"(%1498, %178) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1503 = "pphlo.or"(%1501, %1502) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1504 = "pphlo.xor"(%1499, %1503) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1505 = "pphlo.add"(%1504, %1446) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1506 = "pphlo.add"(%1505, %183) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1507 = "pphlo.add"(%1500, %1506) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1508 = "pphlo.shift_left"(%1506, %176) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1509 = "pphlo.shift_right_logical"(%1506, %175) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1510 = "pphlo.or"(%1508, %1509) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1511 = "pphlo.xor"(%1507, %1510) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1512 = "pphlo.add"(%1507, %1511) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1513 = "pphlo.shift_left"(%1511, %174) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1514 = "pphlo.shift_right_logical"(%1511, %173) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1515 = "pphlo.or"(%1513, %1514) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1516 = "pphlo.xor"(%1512, %1515) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1517 = "pphlo.add"(%1512, %1516) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1518 = "pphlo.shift_left"(%1516, %171) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1519 = "pphlo.shift_right_logical"(%1516, %172) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1520 = "pphlo.or"(%1518, %1519) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1521 = "pphlo.xor"(%1517, %1520) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1522 = "pphlo.add"(%1517, %1521) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1523 = "pphlo.add"(%1522, %1446) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1524 = "pphlo.shift_left"(%1521, %172) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1525 = "pphlo.shift_right_logical"(%1521, %171) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1526 = "pphlo.or"(%1524, %1525) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1527 = "pphlo.xor"(%1522, %1526) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1528 = "pphlo.add"(%1527, %1456) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1529 = "pphlo.add"(%1528, %181) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1530 = "pphlo.add"(%1523, %1529) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1531 = "pphlo.shift_left"(%1529, %173) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1532 = "pphlo.shift_right_logical"(%1529, %174) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1533 = "pphlo.or"(%1531, %1532) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1534 = "pphlo.xor"(%1530, %1533) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1535 = "pphlo.add"(%1530, %1534) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1536 = "pphlo.shift_left"(%1534, %182) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1537 = "pphlo.shift_right_logical"(%1534, %181) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1538 = "pphlo.or"(%1536, %1537) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1539 = "pphlo.xor"(%1535, %1538) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1540 = "pphlo.add"(%1535, %1539) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1541 = "pphlo.shift_left"(%1539, %180) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1542 = "pphlo.shift_right_logical"(%1539, %180) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1543 = "pphlo.or"(%1541, %1542) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1544 = "pphlo.xor"(%1540, %1543) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1545 = "pphlo.add"(%1540, %1544) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1546 = "pphlo.add"(%1545, %1456) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1547 = "pphlo.shift_left"(%1544, %179) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1548 = "pphlo.shift_right_logical"(%1544, %178) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1549 = "pphlo.or"(%1547, %1548) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1550 = "pphlo.xor"(%1545, %1549) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1551 = "pphlo.add"(%1550, %1481) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1552 = "pphlo.add"(%1551, %177) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1553 = "pphlo.add"(%1546, %1552) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1554 = "pphlo.shift_left"(%1552, %176) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1555 = "pphlo.shift_right_logical"(%1552, %175) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1556 = "pphlo.or"(%1554, %1555) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1557 = "pphlo.xor"(%1553, %1556) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1558 = "pphlo.add"(%1553, %1557) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1559 = "pphlo.shift_left"(%1557, %174) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1560 = "pphlo.shift_right_logical"(%1557, %173) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1561 = "pphlo.or"(%1559, %1560) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1562 = "pphlo.xor"(%1558, %1561) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1563 = "pphlo.add"(%1558, %1562) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1564 = "pphlo.shift_left"(%1562, %171) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1565 = "pphlo.shift_right_logical"(%1562, %172) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1566 = "pphlo.or"(%1564, %1565) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1567 = "pphlo.xor"(%1563, %1566) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1568 = "pphlo.add"(%1563, %1567) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1569 = "pphlo.add"(%1568, %1481) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1570 = "pphlo.shift_left"(%1567, %172) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1571 = "pphlo.shift_right_logical"(%1567, %171) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1572 = "pphlo.or"(%1570, %1571) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1573 = "pphlo.xor"(%1568, %1572) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1574 = "pphlo.add"(%1573, %1446) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1575 = "pphlo.add"(%1574, %170) : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<9216x!pphlo.pub<ui32>>
    %1576 = "pphlo.concatenate"(%1569, %1575) {dimension = 0 : i64} : (tensor<9216x!pphlo.pub<ui32>>, tensor<9216x!pphlo.pub<ui32>>) -> tensor<18432x!pphlo.pub<ui32>>
    %1577 = "pphlo.shift_right_logical"(%1576, %169) : (tensor<18432x!pphlo.pub<ui32>>, tensor<18432x!pphlo.pub<ui32>>) -> tensor<18432x!pphlo.pub<ui32>>
    %1578 = "pphlo.or"(%1577, %168) : (tensor<18432x!pphlo.pub<ui32>>, tensor<18432x!pphlo.pub<ui32>>) -> tensor<18432x!pphlo.pub<ui32>>
    %1579 = "pphlo.bitcast_convert"(%1578) {elsize = 32 : i64} : (tensor<18432x!pphlo.pub<ui32>>) -> tensor<18432x!pphlo.pub<f32>>
    %1580 = "pphlo.add"(%1579, %167) : (tensor<18432x!pphlo.pub<f32>>, tensor<18432x!pphlo.pub<f32>>) -> tensor<18432x!pphlo.pub<f32>>
    %1581 = "pphlo.multiply"(%1580, %166) : (tensor<18432x!pphlo.pub<f32>>, tensor<18432x!pphlo.pub<f32>>) -> tensor<18432x!pphlo.pub<f32>>
    %1582 = "pphlo.add"(%1581, %165) : (tensor<18432x!pphlo.pub<f32>>, tensor<18432x!pphlo.pub<f32>>) -> tensor<18432x!pphlo.pub<f32>>
    %1583 = "pphlo.maximum"(%1582, %165) : (tensor<18432x!pphlo.pub<f32>>, tensor<18432x!pphlo.pub<f32>>) -> tensor<18432x!pphlo.pub<f32>>
    %1584 = "pphlo.reshape"(%1583) : (tensor<18432x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1585 = "pphlo.abs"(%1584) : (tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1586 = "pphlo.equal"(%1585, %164) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<i1>>
    %1587 = "pphlo.multiply"(%1584, %163) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1588 = "pphlo.negate"(%1584) : (tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1589 = "pphlo.multiply"(%1588, %1584) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1590 = "pphlo.log_plus_one"(%1589) : (tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1591 = "pphlo.negate"(%1590) : (tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1592 = "pphlo.less"(%1591, %162) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<i1>>
    %1593 = "pphlo.select"(%1592, %161, %160) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1594 = "pphlo.select"(%1592, %159, %158) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1595 = "pphlo.select"(%1592, %157, %156) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1596 = "pphlo.select"(%1592, %155, %154) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1597 = "pphlo.select"(%1592, %153, %152) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1598 = "pphlo.select"(%1592, %151, %150) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1599 = "pphlo.select"(%1592, %149, %148) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1600 = "pphlo.select"(%1592, %147, %146) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1601 = "pphlo.select"(%1592, %145, %144) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1602 = "pphlo.add"(%1591, %143) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1603 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<3x3x32x64xf32>} : () -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1604 = "pphlo.power"(%1591, %1603) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1605 = "pphlo.add"(%1604, %142) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1606 = "pphlo.select"(%1592, %1602, %1605) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1607 = "pphlo.multiply"(%1601, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1608 = "pphlo.add"(%1600, %1607) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1609 = "pphlo.multiply"(%1608, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1610 = "pphlo.add"(%1599, %1609) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1611 = "pphlo.multiply"(%1610, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1612 = "pphlo.add"(%1598, %1611) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1613 = "pphlo.multiply"(%1612, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1614 = "pphlo.add"(%1597, %1613) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1615 = "pphlo.multiply"(%1614, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1616 = "pphlo.add"(%1596, %1615) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1617 = "pphlo.multiply"(%1616, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1618 = "pphlo.add"(%1595, %1617) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1619 = "pphlo.multiply"(%1618, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1620 = "pphlo.add"(%1594, %1619) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1621 = "pphlo.multiply"(%1620, %1606) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1622 = "pphlo.add"(%1593, %1621) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1623 = "pphlo.multiply"(%1622, %1584) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1624 = "pphlo.select"(%1586, %1587, %1623) : (tensor<3x3x32x64x!pphlo.pub<i1>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1625 = "pphlo.multiply"(%1624, %141) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1626 = "pphlo.clamp"(%186, %1625, %140) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1627 = "pphlo.multiply"(%1626, %139) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %1628 = pphlo.convolution(%1211, %1627) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x14x14x32x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<30x14x14x64x!pphlo.pub<f32>>
    %1629 = "pphlo.greater"(%1628, %19) : (tensor<30x14x14x64x!pphlo.pub<f32>>, tensor<30x14x14x64x!pphlo.pub<f32>>) -> tensor<30x14x14x64x!pphlo.pub<i1>>
    %1630 = "pphlo.maximum"(%1628, %19) : (tensor<30x14x14x64x!pphlo.pub<f32>>, tensor<30x14x14x64x!pphlo.pub<f32>>) -> tensor<30x14x14x64x!pphlo.pub<f32>>
    %1631 = "pphlo.reduce_window"(%1630, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<30x14x14x64x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x7x7x64x!pphlo.pub<f32>>
    %1632 = "pphlo.multiply"(%1631, %138) : (tensor<30x7x7x64x!pphlo.pub<f32>>, tensor<30x7x7x64x!pphlo.pub<f32>>) -> tensor<30x7x7x64x!pphlo.pub<f32>>
    %1633 = "pphlo.reshape"(%1632) : (tensor<30x7x7x64x!pphlo.pub<f32>>) -> tensor<30x3136x!pphlo.pub<f32>>
    %1634 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<802816x!pphlo.pub<ui32>>
    %1635 = "pphlo.slice"(%1634) {limit_indices = dense<401408> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<802816x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1636 = "pphlo.add"(%390, %136) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1637 = "pphlo.add"(%387, %1636) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1638 = "pphlo.shift_left"(%1636, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1639 = "pphlo.shift_right_logical"(%1636, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1640 = "pphlo.or"(%1638, %1639) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1641 = "pphlo.xor"(%1637, %1640) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1642 = "pphlo.add"(%1637, %1641) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1643 = "pphlo.shift_left"(%1641, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1644 = "pphlo.shift_right_logical"(%1641, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1645 = "pphlo.or"(%1643, %1644) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1646 = "pphlo.xor"(%1642, %1645) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1647 = "pphlo.add"(%1642, %1646) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1648 = "pphlo.shift_left"(%1646, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1649 = "pphlo.shift_right_logical"(%1646, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1650 = "pphlo.or"(%1648, %1649) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1651 = "pphlo.xor"(%1647, %1650) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1652 = "pphlo.add"(%1647, %1651) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1653 = "pphlo.add"(%1652, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1654 = "pphlo.shift_left"(%1651, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1655 = "pphlo.shift_right_logical"(%1651, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1656 = "pphlo.or"(%1654, %1655) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1657 = "pphlo.xor"(%1652, %1656) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1658 = "pphlo.add"(%1657, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1659 = "pphlo.add"(%1658, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1660 = "pphlo.add"(%1653, %1659) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1661 = "pphlo.shift_left"(%1659, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1662 = "pphlo.shift_right_logical"(%1659, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1663 = "pphlo.or"(%1661, %1662) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1664 = "pphlo.xor"(%1660, %1663) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1665 = "pphlo.add"(%1660, %1664) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1666 = "pphlo.shift_left"(%1664, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1667 = "pphlo.shift_right_logical"(%1664, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1668 = "pphlo.or"(%1666, %1667) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1669 = "pphlo.xor"(%1665, %1668) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1670 = "pphlo.add"(%1665, %1669) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1671 = "pphlo.shift_left"(%1669, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1672 = "pphlo.shift_right_logical"(%1669, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1673 = "pphlo.or"(%1671, %1672) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1674 = "pphlo.xor"(%1670, %1673) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1675 = "pphlo.add"(%1670, %1674) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1676 = "pphlo.add"(%1675, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1677 = "pphlo.shift_left"(%1674, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1678 = "pphlo.shift_right_logical"(%1674, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1679 = "pphlo.or"(%1677, %1678) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1680 = "pphlo.xor"(%1675, %1679) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1681 = "pphlo.add"(%1680, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1682 = "pphlo.add"(%1681, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1683 = "pphlo.add"(%1676, %1682) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1684 = "pphlo.shift_left"(%1682, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1685 = "pphlo.shift_right_logical"(%1682, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1686 = "pphlo.or"(%1684, %1685) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1687 = "pphlo.xor"(%1683, %1686) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1688 = "pphlo.add"(%1683, %1687) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1689 = "pphlo.shift_left"(%1687, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1690 = "pphlo.shift_right_logical"(%1687, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1691 = "pphlo.or"(%1689, %1690) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1692 = "pphlo.xor"(%1688, %1691) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1693 = "pphlo.add"(%1688, %1692) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1694 = "pphlo.shift_left"(%1692, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1695 = "pphlo.shift_right_logical"(%1692, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1696 = "pphlo.or"(%1694, %1695) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1697 = "pphlo.xor"(%1693, %1696) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1698 = "pphlo.add"(%1693, %1697) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1699 = "pphlo.add"(%1698, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1700 = "pphlo.shift_left"(%1697, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1701 = "pphlo.shift_right_logical"(%1697, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1702 = "pphlo.or"(%1700, %1701) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1703 = "pphlo.xor"(%1698, %1702) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1704 = "pphlo.add"(%1703, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1705 = "pphlo.add"(%1704, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1706 = "pphlo.add"(%1699, %1705) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1707 = "pphlo.shift_left"(%1705, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1708 = "pphlo.shift_right_logical"(%1705, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1709 = "pphlo.or"(%1707, %1708) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1710 = "pphlo.xor"(%1706, %1709) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1711 = "pphlo.add"(%1706, %1710) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1712 = "pphlo.shift_left"(%1710, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1713 = "pphlo.shift_right_logical"(%1710, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1714 = "pphlo.or"(%1712, %1713) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1715 = "pphlo.xor"(%1711, %1714) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1716 = "pphlo.add"(%1711, %1715) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1717 = "pphlo.shift_left"(%1715, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1718 = "pphlo.shift_right_logical"(%1715, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1719 = "pphlo.or"(%1717, %1718) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1720 = "pphlo.xor"(%1716, %1719) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1721 = "pphlo.add"(%1716, %1720) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1722 = "pphlo.add"(%1721, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1723 = "pphlo.shift_left"(%1720, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1724 = "pphlo.shift_right_logical"(%1720, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1725 = "pphlo.or"(%1723, %1724) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1726 = "pphlo.xor"(%1721, %1725) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1727 = "pphlo.add"(%1726, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1728 = "pphlo.add"(%1727, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1729 = "pphlo.add"(%1722, %1728) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1730 = "pphlo.shift_left"(%1728, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1731 = "pphlo.shift_right_logical"(%1728, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1732 = "pphlo.or"(%1730, %1731) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1733 = "pphlo.xor"(%1729, %1732) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1734 = "pphlo.add"(%1729, %1733) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1735 = "pphlo.shift_left"(%1733, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1736 = "pphlo.shift_right_logical"(%1733, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1737 = "pphlo.or"(%1735, %1736) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1738 = "pphlo.xor"(%1734, %1737) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1739 = "pphlo.add"(%1734, %1738) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1740 = "pphlo.shift_left"(%1738, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1741 = "pphlo.shift_right_logical"(%1738, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1742 = "pphlo.or"(%1740, %1741) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1743 = "pphlo.xor"(%1739, %1742) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1744 = "pphlo.add"(%1739, %1743) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1745 = "pphlo.add"(%1744, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1746 = "pphlo.shift_left"(%1743, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1747 = "pphlo.shift_right_logical"(%1743, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1748 = "pphlo.or"(%1746, %1747) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1749 = "pphlo.xor"(%1744, %1748) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1750 = "pphlo.add"(%1749, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1751 = "pphlo.add"(%1750, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1752 = "pphlo.add"(%1745, %1751) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1753 = "pphlo.shift_left"(%1751, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1754 = "pphlo.shift_right_logical"(%1751, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1755 = "pphlo.or"(%1753, %1754) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1756 = "pphlo.xor"(%1752, %1755) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1757 = "pphlo.add"(%1752, %1756) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1758 = "pphlo.shift_left"(%1756, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1759 = "pphlo.shift_right_logical"(%1756, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1760 = "pphlo.or"(%1758, %1759) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1761 = "pphlo.xor"(%1757, %1760) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1762 = "pphlo.add"(%1757, %1761) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1763 = "pphlo.shift_left"(%1761, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1764 = "pphlo.shift_right_logical"(%1761, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1765 = "pphlo.or"(%1763, %1764) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1766 = "pphlo.xor"(%1762, %1765) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1767 = "pphlo.add"(%1762, %1766) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1768 = "pphlo.add"(%1750, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1769 = "pphlo.add"(%1767, %1768) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1770 = "pphlo.shift_left"(%1766, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1771 = "pphlo.shift_right_logical"(%1766, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1772 = "pphlo.or"(%1770, %1771) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1773 = "pphlo.xor"(%1767, %1772) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1774 = "pphlo.reshape"(%1745) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1775 = "pphlo.reshape"(%1768) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1776 = "pphlo.xor"(%1774, %1775) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1777 = "pphlo.xor"(%1776, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1778 = "pphlo.reshape"(%1777) : (tensor<!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1779 = "pphlo.add"(%1773, %1778) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1780 = "pphlo.add"(%1779, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1781 = "pphlo.add"(%1769, %1780) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1782 = "pphlo.shift_left"(%1780, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1783 = "pphlo.shift_right_logical"(%1780, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1784 = "pphlo.or"(%1782, %1783) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1785 = "pphlo.xor"(%1781, %1784) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1786 = "pphlo.add"(%1781, %1785) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1787 = "pphlo.shift_left"(%1785, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1788 = "pphlo.shift_right_logical"(%1785, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1789 = "pphlo.or"(%1787, %1788) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1790 = "pphlo.xor"(%1786, %1789) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1791 = "pphlo.add"(%1786, %1790) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1792 = "pphlo.shift_left"(%1790, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1793 = "pphlo.shift_right_logical"(%1790, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1794 = "pphlo.or"(%1792, %1793) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1795 = "pphlo.xor"(%1791, %1794) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1796 = "pphlo.add"(%1791, %1795) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1797 = "pphlo.add"(%1796, %1778) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1798 = "pphlo.shift_left"(%1795, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1799 = "pphlo.shift_right_logical"(%1795, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1800 = "pphlo.or"(%1798, %1799) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1801 = "pphlo.xor"(%1796, %1800) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1802 = "pphlo.add"(%1801, %1745) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1803 = "pphlo.add"(%1802, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1804 = "pphlo.add"(%1797, %1803) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1805 = "pphlo.shift_left"(%1803, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1806 = "pphlo.shift_right_logical"(%1803, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1807 = "pphlo.or"(%1805, %1806) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1808 = "pphlo.xor"(%1804, %1807) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1809 = "pphlo.add"(%1804, %1808) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1810 = "pphlo.shift_left"(%1808, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1811 = "pphlo.shift_right_logical"(%1808, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1812 = "pphlo.or"(%1810, %1811) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1813 = "pphlo.xor"(%1809, %1812) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1814 = "pphlo.add"(%1809, %1813) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1815 = "pphlo.shift_left"(%1813, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1816 = "pphlo.shift_right_logical"(%1813, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1817 = "pphlo.or"(%1815, %1816) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1818 = "pphlo.xor"(%1814, %1817) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1819 = "pphlo.add"(%1814, %1818) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1820 = "pphlo.add"(%1819, %1745) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1821 = "pphlo.shift_left"(%1818, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1822 = "pphlo.shift_right_logical"(%1818, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1823 = "pphlo.or"(%1821, %1822) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1824 = "pphlo.xor"(%1819, %1823) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1825 = "pphlo.add"(%1824, %1768) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1826 = "pphlo.add"(%1825, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1827 = "pphlo.add"(%1820, %1826) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1828 = "pphlo.shift_left"(%1826, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1829 = "pphlo.shift_right_logical"(%1826, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1830 = "pphlo.or"(%1828, %1829) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1831 = "pphlo.xor"(%1827, %1830) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1832 = "pphlo.add"(%1827, %1831) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1833 = "pphlo.shift_left"(%1831, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1834 = "pphlo.shift_right_logical"(%1831, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1835 = "pphlo.or"(%1833, %1834) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1836 = "pphlo.xor"(%1832, %1835) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1837 = "pphlo.add"(%1832, %1836) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1838 = "pphlo.shift_left"(%1836, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1839 = "pphlo.shift_right_logical"(%1836, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1840 = "pphlo.or"(%1838, %1839) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1841 = "pphlo.xor"(%1837, %1840) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1842 = "pphlo.add"(%1837, %1841) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1843 = "pphlo.add"(%1842, %1768) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1844 = "pphlo.shift_left"(%1841, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1845 = "pphlo.shift_right_logical"(%1841, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1846 = "pphlo.or"(%1844, %1845) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1847 = "pphlo.xor"(%1842, %1846) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1848 = "pphlo.add"(%1847, %1778) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1849 = "pphlo.add"(%1848, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1850 = "pphlo.add"(%1843, %1849) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1851 = "pphlo.shift_left"(%1849, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1852 = "pphlo.shift_right_logical"(%1849, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1853 = "pphlo.or"(%1851, %1852) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1854 = "pphlo.xor"(%1850, %1853) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1855 = "pphlo.add"(%1850, %1854) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1856 = "pphlo.shift_left"(%1854, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1857 = "pphlo.shift_right_logical"(%1854, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1858 = "pphlo.or"(%1856, %1857) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1859 = "pphlo.xor"(%1855, %1858) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1860 = "pphlo.add"(%1855, %1859) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1861 = "pphlo.shift_left"(%1859, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1862 = "pphlo.shift_right_logical"(%1859, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1863 = "pphlo.or"(%1861, %1862) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1864 = "pphlo.xor"(%1860, %1863) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1865 = "pphlo.add"(%1860, %1864) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1866 = "pphlo.add"(%1865, %1778) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1867 = "pphlo.reshape"(%1866) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1868 = "pphlo.broadcast"(%1867) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1869 = "pphlo.add"(%1635, %1868) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1870 = "pphlo.slice"(%1634) {limit_indices = dense<802816> : tensor<1xi64>, start_indices = dense<401408> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<802816x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1871 = "pphlo.shift_left"(%1864, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1872 = "pphlo.shift_right_logical"(%1864, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1873 = "pphlo.or"(%1871, %1872) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1874 = "pphlo.xor"(%1865, %1873) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1875 = "pphlo.add"(%1874, %1745) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1876 = "pphlo.add"(%1875, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %1877 = "pphlo.reshape"(%1876) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1878 = "pphlo.broadcast"(%1877) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1879 = "pphlo.add"(%1870, %1878) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1880 = "pphlo.add"(%1869, %1879) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1881 = "pphlo.shift_left"(%1879, %127) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1882 = "pphlo.shift_right_logical"(%1879, %126) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1883 = "pphlo.or"(%1881, %1882) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1884 = "pphlo.xor"(%1880, %1883) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1885 = "pphlo.add"(%1880, %1884) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1886 = "pphlo.shift_left"(%1884, %125) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1887 = "pphlo.shift_right_logical"(%1884, %124) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1888 = "pphlo.or"(%1886, %1887) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1889 = "pphlo.xor"(%1885, %1888) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1890 = "pphlo.add"(%1885, %1889) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1891 = "pphlo.shift_left"(%1889, %122) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1892 = "pphlo.shift_right_logical"(%1889, %123) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1893 = "pphlo.or"(%1891, %1892) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1894 = "pphlo.xor"(%1890, %1893) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1895 = "pphlo.add"(%1890, %1894) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1896 = "pphlo.add"(%1895, %1878) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1897 = "pphlo.shift_left"(%1894, %123) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1898 = "pphlo.shift_right_logical"(%1894, %122) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1899 = "pphlo.or"(%1897, %1898) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1900 = "pphlo.xor"(%1895, %1899) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1901 = "pphlo.xor"(%1867, %1877) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1902 = "pphlo.xor"(%1901, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %1903 = "pphlo.broadcast"(%1902) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1904 = "pphlo.add"(%1900, %1903) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1905 = "pphlo.add"(%1904, %135) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1906 = "pphlo.add"(%1896, %1905) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1907 = "pphlo.shift_left"(%1905, %124) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1908 = "pphlo.shift_right_logical"(%1905, %125) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1909 = "pphlo.or"(%1907, %1908) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1910 = "pphlo.xor"(%1906, %1909) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1911 = "pphlo.add"(%1906, %1910) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1912 = "pphlo.shift_left"(%1910, %133) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1913 = "pphlo.shift_right_logical"(%1910, %132) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1914 = "pphlo.or"(%1912, %1913) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1915 = "pphlo.xor"(%1911, %1914) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1916 = "pphlo.add"(%1911, %1915) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1917 = "pphlo.shift_left"(%1915, %131) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1918 = "pphlo.shift_right_logical"(%1915, %131) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1919 = "pphlo.or"(%1917, %1918) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1920 = "pphlo.xor"(%1916, %1919) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1921 = "pphlo.add"(%1916, %1920) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1922 = "pphlo.add"(%1921, %1903) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1923 = "pphlo.shift_left"(%1920, %130) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1924 = "pphlo.shift_right_logical"(%1920, %129) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1925 = "pphlo.or"(%1923, %1924) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1926 = "pphlo.xor"(%1921, %1925) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1927 = "pphlo.add"(%1926, %1868) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1928 = "pphlo.add"(%1927, %134) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1929 = "pphlo.add"(%1922, %1928) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1930 = "pphlo.shift_left"(%1928, %127) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1931 = "pphlo.shift_right_logical"(%1928, %126) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1932 = "pphlo.or"(%1930, %1931) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1933 = "pphlo.xor"(%1929, %1932) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1934 = "pphlo.add"(%1929, %1933) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1935 = "pphlo.shift_left"(%1933, %125) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1936 = "pphlo.shift_right_logical"(%1933, %124) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1937 = "pphlo.or"(%1935, %1936) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1938 = "pphlo.xor"(%1934, %1937) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1939 = "pphlo.add"(%1934, %1938) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1940 = "pphlo.shift_left"(%1938, %122) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1941 = "pphlo.shift_right_logical"(%1938, %123) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1942 = "pphlo.or"(%1940, %1941) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1943 = "pphlo.xor"(%1939, %1942) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1944 = "pphlo.add"(%1939, %1943) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1945 = "pphlo.add"(%1944, %1868) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1946 = "pphlo.shift_left"(%1943, %123) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1947 = "pphlo.shift_right_logical"(%1943, %122) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1948 = "pphlo.or"(%1946, %1947) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1949 = "pphlo.xor"(%1944, %1948) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1950 = "pphlo.add"(%1949, %1878) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1951 = "pphlo.add"(%1950, %132) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1952 = "pphlo.add"(%1945, %1951) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1953 = "pphlo.shift_left"(%1951, %124) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1954 = "pphlo.shift_right_logical"(%1951, %125) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1955 = "pphlo.or"(%1953, %1954) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1956 = "pphlo.xor"(%1952, %1955) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1957 = "pphlo.add"(%1952, %1956) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1958 = "pphlo.shift_left"(%1956, %133) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1959 = "pphlo.shift_right_logical"(%1956, %132) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1960 = "pphlo.or"(%1958, %1959) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1961 = "pphlo.xor"(%1957, %1960) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1962 = "pphlo.add"(%1957, %1961) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1963 = "pphlo.shift_left"(%1961, %131) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1964 = "pphlo.shift_right_logical"(%1961, %131) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1965 = "pphlo.or"(%1963, %1964) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1966 = "pphlo.xor"(%1962, %1965) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1967 = "pphlo.add"(%1962, %1966) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1968 = "pphlo.add"(%1967, %1878) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1969 = "pphlo.shift_left"(%1966, %130) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1970 = "pphlo.shift_right_logical"(%1966, %129) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1971 = "pphlo.or"(%1969, %1970) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1972 = "pphlo.xor"(%1967, %1971) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1973 = "pphlo.add"(%1972, %1903) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1974 = "pphlo.add"(%1973, %128) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1975 = "pphlo.add"(%1968, %1974) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1976 = "pphlo.shift_left"(%1974, %127) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1977 = "pphlo.shift_right_logical"(%1974, %126) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1978 = "pphlo.or"(%1976, %1977) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1979 = "pphlo.xor"(%1975, %1978) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1980 = "pphlo.add"(%1975, %1979) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1981 = "pphlo.shift_left"(%1979, %125) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1982 = "pphlo.shift_right_logical"(%1979, %124) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1983 = "pphlo.or"(%1981, %1982) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1984 = "pphlo.xor"(%1980, %1983) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1985 = "pphlo.add"(%1980, %1984) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1986 = "pphlo.shift_left"(%1984, %122) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1987 = "pphlo.shift_right_logical"(%1984, %123) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1988 = "pphlo.or"(%1986, %1987) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1989 = "pphlo.xor"(%1985, %1988) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1990 = "pphlo.add"(%1985, %1989) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1991 = "pphlo.add"(%1990, %1903) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1992 = "pphlo.shift_left"(%1989, %123) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1993 = "pphlo.shift_right_logical"(%1989, %122) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1994 = "pphlo.or"(%1992, %1993) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1995 = "pphlo.xor"(%1990, %1994) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1996 = "pphlo.add"(%1995, %1868) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1997 = "pphlo.add"(%1996, %121) : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<401408x!pphlo.pub<ui32>>
    %1998 = "pphlo.concatenate"(%1991, %1997) {dimension = 0 : i64} : (tensor<401408x!pphlo.pub<ui32>>, tensor<401408x!pphlo.pub<ui32>>) -> tensor<802816x!pphlo.pub<ui32>>
    %1999 = "pphlo.shift_right_logical"(%1998, %120) : (tensor<802816x!pphlo.pub<ui32>>, tensor<802816x!pphlo.pub<ui32>>) -> tensor<802816x!pphlo.pub<ui32>>
    %2000 = "pphlo.or"(%1999, %119) : (tensor<802816x!pphlo.pub<ui32>>, tensor<802816x!pphlo.pub<ui32>>) -> tensor<802816x!pphlo.pub<ui32>>
    %2001 = "pphlo.bitcast_convert"(%2000) {elsize = 32 : i64} : (tensor<802816x!pphlo.pub<ui32>>) -> tensor<802816x!pphlo.pub<f32>>
    %2002 = "pphlo.add"(%2001, %118) : (tensor<802816x!pphlo.pub<f32>>, tensor<802816x!pphlo.pub<f32>>) -> tensor<802816x!pphlo.pub<f32>>
    %2003 = "pphlo.multiply"(%2002, %117) : (tensor<802816x!pphlo.pub<f32>>, tensor<802816x!pphlo.pub<f32>>) -> tensor<802816x!pphlo.pub<f32>>
    %2004 = "pphlo.add"(%2003, %116) : (tensor<802816x!pphlo.pub<f32>>, tensor<802816x!pphlo.pub<f32>>) -> tensor<802816x!pphlo.pub<f32>>
    %2005 = "pphlo.maximum"(%2004, %116) : (tensor<802816x!pphlo.pub<f32>>, tensor<802816x!pphlo.pub<f32>>) -> tensor<802816x!pphlo.pub<f32>>
    %2006 = "pphlo.reshape"(%2005) : (tensor<802816x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2007 = "pphlo.abs"(%2006) : (tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2008 = "pphlo.equal"(%2007, %115) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<i1>>
    %2009 = "pphlo.multiply"(%2006, %114) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2010 = "pphlo.negate"(%2006) : (tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2011 = "pphlo.multiply"(%2010, %2006) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2012 = "pphlo.log_plus_one"(%2011) : (tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2013 = "pphlo.negate"(%2012) : (tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2014 = "pphlo.less"(%2013, %113) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<i1>>
    %2015 = "pphlo.select"(%2014, %112, %111) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2016 = "pphlo.select"(%2014, %110, %109) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2017 = "pphlo.select"(%2014, %108, %107) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2018 = "pphlo.select"(%2014, %106, %105) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2019 = "pphlo.select"(%2014, %104, %103) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2020 = "pphlo.select"(%2014, %102, %101) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2021 = "pphlo.select"(%2014, %100, %99) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2022 = "pphlo.select"(%2014, %98, %97) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2023 = "pphlo.select"(%2014, %96, %95) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2024 = "pphlo.add"(%2013, %94) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2025 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<3136x256xf32>} : () -> tensor<3136x256x!pphlo.pub<f32>>
    %2026 = "pphlo.power"(%2013, %2025) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2027 = "pphlo.add"(%2026, %93) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2028 = "pphlo.select"(%2014, %2024, %2027) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2029 = "pphlo.multiply"(%2023, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2030 = "pphlo.add"(%2022, %2029) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2031 = "pphlo.multiply"(%2030, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2032 = "pphlo.add"(%2021, %2031) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2033 = "pphlo.multiply"(%2032, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2034 = "pphlo.add"(%2020, %2033) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2035 = "pphlo.multiply"(%2034, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2036 = "pphlo.add"(%2019, %2035) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2037 = "pphlo.multiply"(%2036, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2038 = "pphlo.add"(%2018, %2037) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2039 = "pphlo.multiply"(%2038, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2040 = "pphlo.add"(%2017, %2039) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2041 = "pphlo.multiply"(%2040, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2042 = "pphlo.add"(%2016, %2041) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2043 = "pphlo.multiply"(%2042, %2028) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2044 = "pphlo.add"(%2015, %2043) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2045 = "pphlo.multiply"(%2044, %2006) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2046 = "pphlo.select"(%2008, %2009, %2045) : (tensor<3136x256x!pphlo.pub<i1>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2047 = "pphlo.multiply"(%2046, %92) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2048 = "pphlo.clamp"(%137, %2047, %91) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2049 = "pphlo.multiply"(%2048, %90) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2050 = "pphlo.dot"(%1633, %2049) : (tensor<30x3136x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<30x256x!pphlo.pub<f32>>
    %2051 = "pphlo.greater"(%2050, %21) : (tensor<30x256x!pphlo.pub<f32>>, tensor<30x256x!pphlo.pub<f32>>) -> tensor<30x256x!pphlo.pub<i1>>
    %2052 = "pphlo.gather"(%arg1, %1205) {dimension_numbers = #pphlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<30x!pphlo.pub<i32>>, tensor<30x1x!pphlo.pub<i32>>) -> tensor<30x!pphlo.pub<i32>>
    %2053 = "pphlo.broadcast"(%2052) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<30x!pphlo.pub<i32>>) -> tensor<30x10x!pphlo.pub<i32>>
    %2054 = "pphlo.broadcast"(%269) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10x!pphlo.pub<i32>>) -> tensor<30x10x!pphlo.pub<i32>>
    %2055 = "pphlo.equal"(%2053, %2054) : (tensor<30x10x!pphlo.pub<i32>>, tensor<30x10x!pphlo.pub<i32>>) -> tensor<30x10x!pphlo.pub<i1>>
    %2056 = "pphlo.select"(%2055, %87, %86) : (tensor<30x10x!pphlo.pub<i1>>, tensor<30x10x!pphlo.pub<f32>>, tensor<30x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2057 = "pphlo.negate"(%2056) : (tensor<30x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2058 = "pphlo.reduce"(%2057, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<30x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x!pphlo.pub<f32>>
    %2059 = "pphlo.maximum"(%2050, %21) : (tensor<30x256x!pphlo.pub<f32>>, tensor<30x256x!pphlo.pub<f32>>) -> tensor<30x256x!pphlo.pub<f32>>
    %2060 = "pphlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2560x!pphlo.pub<ui32>>
    %2061 = "pphlo.slice"(%2060) {limit_indices = dense<1280> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2560x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2062 = "pphlo.add"(%390, %84) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2063 = "pphlo.add"(%387, %2062) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2064 = "pphlo.shift_left"(%2062, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2065 = "pphlo.shift_right_logical"(%2062, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2066 = "pphlo.or"(%2064, %2065) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2067 = "pphlo.xor"(%2063, %2066) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2068 = "pphlo.add"(%2063, %2067) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2069 = "pphlo.shift_left"(%2067, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2070 = "pphlo.shift_right_logical"(%2067, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2071 = "pphlo.or"(%2069, %2070) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2072 = "pphlo.xor"(%2068, %2071) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2073 = "pphlo.add"(%2068, %2072) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2074 = "pphlo.shift_left"(%2072, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2075 = "pphlo.shift_right_logical"(%2072, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2076 = "pphlo.or"(%2074, %2075) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2077 = "pphlo.xor"(%2073, %2076) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2078 = "pphlo.add"(%2073, %2077) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2079 = "pphlo.add"(%2078, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2080 = "pphlo.shift_left"(%2077, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2081 = "pphlo.shift_right_logical"(%2077, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2082 = "pphlo.or"(%2080, %2081) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2083 = "pphlo.xor"(%2078, %2082) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2084 = "pphlo.add"(%2083, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2085 = "pphlo.add"(%2084, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2086 = "pphlo.add"(%2079, %2085) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2087 = "pphlo.shift_left"(%2085, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2088 = "pphlo.shift_right_logical"(%2085, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2089 = "pphlo.or"(%2087, %2088) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2090 = "pphlo.xor"(%2086, %2089) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2091 = "pphlo.add"(%2086, %2090) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2092 = "pphlo.shift_left"(%2090, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2093 = "pphlo.shift_right_logical"(%2090, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2094 = "pphlo.or"(%2092, %2093) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2095 = "pphlo.xor"(%2091, %2094) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2096 = "pphlo.add"(%2091, %2095) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2097 = "pphlo.shift_left"(%2095, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2098 = "pphlo.shift_right_logical"(%2095, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2099 = "pphlo.or"(%2097, %2098) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2100 = "pphlo.xor"(%2096, %2099) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2101 = "pphlo.add"(%2096, %2100) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2102 = "pphlo.add"(%2101, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2103 = "pphlo.shift_left"(%2100, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2104 = "pphlo.shift_right_logical"(%2100, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2105 = "pphlo.or"(%2103, %2104) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2106 = "pphlo.xor"(%2101, %2105) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2107 = "pphlo.add"(%2106, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2108 = "pphlo.add"(%2107, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2109 = "pphlo.add"(%2102, %2108) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2110 = "pphlo.shift_left"(%2108, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2111 = "pphlo.shift_right_logical"(%2108, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2112 = "pphlo.or"(%2110, %2111) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2113 = "pphlo.xor"(%2109, %2112) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2114 = "pphlo.add"(%2109, %2113) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2115 = "pphlo.shift_left"(%2113, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2116 = "pphlo.shift_right_logical"(%2113, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2117 = "pphlo.or"(%2115, %2116) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2118 = "pphlo.xor"(%2114, %2117) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2119 = "pphlo.add"(%2114, %2118) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2120 = "pphlo.shift_left"(%2118, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2121 = "pphlo.shift_right_logical"(%2118, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2122 = "pphlo.or"(%2120, %2121) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2123 = "pphlo.xor"(%2119, %2122) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2124 = "pphlo.add"(%2119, %2123) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2125 = "pphlo.add"(%2124, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2126 = "pphlo.shift_left"(%2123, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2127 = "pphlo.shift_right_logical"(%2123, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2128 = "pphlo.or"(%2126, %2127) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2129 = "pphlo.xor"(%2124, %2128) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2130 = "pphlo.add"(%2129, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2131 = "pphlo.add"(%2130, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2132 = "pphlo.add"(%2125, %2131) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2133 = "pphlo.shift_left"(%2131, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2134 = "pphlo.shift_right_logical"(%2131, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2135 = "pphlo.or"(%2133, %2134) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2136 = "pphlo.xor"(%2132, %2135) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2137 = "pphlo.add"(%2132, %2136) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2138 = "pphlo.shift_left"(%2136, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2139 = "pphlo.shift_right_logical"(%2136, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2140 = "pphlo.or"(%2138, %2139) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2141 = "pphlo.xor"(%2137, %2140) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2142 = "pphlo.add"(%2137, %2141) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2143 = "pphlo.shift_left"(%2141, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2144 = "pphlo.shift_right_logical"(%2141, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2145 = "pphlo.or"(%2143, %2144) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2146 = "pphlo.xor"(%2142, %2145) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2147 = "pphlo.add"(%2142, %2146) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2148 = "pphlo.add"(%2147, %390) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2149 = "pphlo.shift_left"(%2146, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2150 = "pphlo.shift_right_logical"(%2146, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2151 = "pphlo.or"(%2149, %2150) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2152 = "pphlo.xor"(%2147, %2151) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2153 = "pphlo.add"(%2152, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2154 = "pphlo.add"(%2153, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2155 = "pphlo.add"(%2148, %2154) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2156 = "pphlo.shift_left"(%2154, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2157 = "pphlo.shift_right_logical"(%2154, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2158 = "pphlo.or"(%2156, %2157) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2159 = "pphlo.xor"(%2155, %2158) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2160 = "pphlo.add"(%2155, %2159) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2161 = "pphlo.shift_left"(%2159, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2162 = "pphlo.shift_right_logical"(%2159, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2163 = "pphlo.or"(%2161, %2162) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2164 = "pphlo.xor"(%2160, %2163) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2165 = "pphlo.add"(%2160, %2164) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2166 = "pphlo.shift_left"(%2164, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2167 = "pphlo.shift_right_logical"(%2164, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2168 = "pphlo.or"(%2166, %2167) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2169 = "pphlo.xor"(%2165, %2168) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2170 = "pphlo.add"(%2165, %2169) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2171 = "pphlo.add"(%2170, %417) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2172 = "pphlo.shift_left"(%2169, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2173 = "pphlo.shift_right_logical"(%2169, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2174 = "pphlo.or"(%2172, %2173) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2175 = "pphlo.xor"(%2170, %2174) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2176 = "pphlo.add"(%2175, %387) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2177 = "pphlo.add"(%2176, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2178 = "pphlo.add"(%2171, %2177) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2179 = "pphlo.shift_left"(%2177, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2180 = "pphlo.shift_right_logical"(%2177, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2181 = "pphlo.or"(%2179, %2180) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2182 = "pphlo.xor"(%2178, %2181) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2183 = "pphlo.add"(%2178, %2182) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2184 = "pphlo.shift_left"(%2182, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2185 = "pphlo.shift_right_logical"(%2182, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2186 = "pphlo.or"(%2184, %2185) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2187 = "pphlo.xor"(%2183, %2186) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2188 = "pphlo.add"(%2183, %2187) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2189 = "pphlo.shift_left"(%2187, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2190 = "pphlo.shift_right_logical"(%2187, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2191 = "pphlo.or"(%2189, %2190) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2192 = "pphlo.xor"(%2188, %2191) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2193 = "pphlo.add"(%2188, %2192) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2194 = "pphlo.add"(%2176, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2195 = "pphlo.add"(%2193, %2194) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2196 = "pphlo.shift_left"(%2192, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2197 = "pphlo.shift_right_logical"(%2192, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2198 = "pphlo.or"(%2196, %2197) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2199 = "pphlo.xor"(%2193, %2198) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2200 = "pphlo.reshape"(%2171) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2201 = "pphlo.reshape"(%2194) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2202 = "pphlo.xor"(%2200, %2201) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2203 = "pphlo.xor"(%2202, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2204 = "pphlo.reshape"(%2203) : (tensor<!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2205 = "pphlo.add"(%2199, %2204) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2206 = "pphlo.add"(%2205, %83) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2207 = "pphlo.add"(%2195, %2206) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2208 = "pphlo.shift_left"(%2206, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2209 = "pphlo.shift_right_logical"(%2206, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2210 = "pphlo.or"(%2208, %2209) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2211 = "pphlo.xor"(%2207, %2210) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2212 = "pphlo.add"(%2207, %2211) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2213 = "pphlo.shift_left"(%2211, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2214 = "pphlo.shift_right_logical"(%2211, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2215 = "pphlo.or"(%2213, %2214) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2216 = "pphlo.xor"(%2212, %2215) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2217 = "pphlo.add"(%2212, %2216) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2218 = "pphlo.shift_left"(%2216, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2219 = "pphlo.shift_right_logical"(%2216, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2220 = "pphlo.or"(%2218, %2219) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2221 = "pphlo.xor"(%2217, %2220) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2222 = "pphlo.add"(%2217, %2221) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2223 = "pphlo.add"(%2222, %2204) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2224 = "pphlo.shift_left"(%2221, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2225 = "pphlo.shift_right_logical"(%2221, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2226 = "pphlo.or"(%2224, %2225) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2227 = "pphlo.xor"(%2222, %2226) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2228 = "pphlo.add"(%2227, %2171) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2229 = "pphlo.add"(%2228, %82) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2230 = "pphlo.add"(%2223, %2229) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2231 = "pphlo.shift_left"(%2229, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2232 = "pphlo.shift_right_logical"(%2229, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2233 = "pphlo.or"(%2231, %2232) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2234 = "pphlo.xor"(%2230, %2233) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2235 = "pphlo.add"(%2230, %2234) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2236 = "pphlo.shift_left"(%2234, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2237 = "pphlo.shift_right_logical"(%2234, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2238 = "pphlo.or"(%2236, %2237) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2239 = "pphlo.xor"(%2235, %2238) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2240 = "pphlo.add"(%2235, %2239) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2241 = "pphlo.shift_left"(%2239, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2242 = "pphlo.shift_right_logical"(%2239, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2243 = "pphlo.or"(%2241, %2242) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2244 = "pphlo.xor"(%2240, %2243) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2245 = "pphlo.add"(%2240, %2244) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2246 = "pphlo.add"(%2245, %2171) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2247 = "pphlo.shift_left"(%2244, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2248 = "pphlo.shift_right_logical"(%2244, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2249 = "pphlo.or"(%2247, %2248) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2250 = "pphlo.xor"(%2245, %2249) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2251 = "pphlo.add"(%2250, %2194) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2252 = "pphlo.add"(%2251, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2253 = "pphlo.add"(%2246, %2252) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2254 = "pphlo.shift_left"(%2252, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2255 = "pphlo.shift_right_logical"(%2252, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2256 = "pphlo.or"(%2254, %2255) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2257 = "pphlo.xor"(%2253, %2256) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2258 = "pphlo.add"(%2253, %2257) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2259 = "pphlo.shift_left"(%2257, %81) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2260 = "pphlo.shift_right_logical"(%2257, %80) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2261 = "pphlo.or"(%2259, %2260) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2262 = "pphlo.xor"(%2258, %2261) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2263 = "pphlo.add"(%2258, %2262) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2264 = "pphlo.shift_left"(%2262, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2265 = "pphlo.shift_right_logical"(%2262, %79) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2266 = "pphlo.or"(%2264, %2265) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2267 = "pphlo.xor"(%2263, %2266) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2268 = "pphlo.add"(%2263, %2267) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2269 = "pphlo.add"(%2268, %2194) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2270 = "pphlo.shift_left"(%2267, %78) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2271 = "pphlo.shift_right_logical"(%2267, %77) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2272 = "pphlo.or"(%2270, %2271) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2273 = "pphlo.xor"(%2268, %2272) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2274 = "pphlo.add"(%2273, %2204) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2275 = "pphlo.add"(%2274, %76) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2276 = "pphlo.add"(%2269, %2275) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2277 = "pphlo.shift_left"(%2275, %75) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2278 = "pphlo.shift_right_logical"(%2275, %74) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2279 = "pphlo.or"(%2277, %2278) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2280 = "pphlo.xor"(%2276, %2279) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2281 = "pphlo.add"(%2276, %2280) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2282 = "pphlo.shift_left"(%2280, %73) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2283 = "pphlo.shift_right_logical"(%2280, %72) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2284 = "pphlo.or"(%2282, %2283) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2285 = "pphlo.xor"(%2281, %2284) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2286 = "pphlo.add"(%2281, %2285) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2287 = "pphlo.shift_left"(%2285, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2288 = "pphlo.shift_right_logical"(%2285, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2289 = "pphlo.or"(%2287, %2288) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2290 = "pphlo.xor"(%2286, %2289) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2291 = "pphlo.add"(%2286, %2290) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2292 = "pphlo.add"(%2291, %2204) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2293 = "pphlo.reshape"(%2292) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2294 = "pphlo.broadcast"(%2293) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2295 = "pphlo.add"(%2061, %2294) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2296 = "pphlo.slice"(%2060) {limit_indices = dense<2560> : tensor<1xi64>, start_indices = dense<1280> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2560x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2297 = "pphlo.shift_left"(%2290, %71) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2298 = "pphlo.shift_right_logical"(%2290, %70) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2299 = "pphlo.or"(%2297, %2298) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2300 = "pphlo.xor"(%2291, %2299) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2301 = "pphlo.add"(%2300, %2171) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2302 = "pphlo.add"(%2301, %69) : (tensor<1x!pphlo.pub<ui32>>, tensor<1x!pphlo.pub<ui32>>) -> tensor<1x!pphlo.pub<ui32>>
    %2303 = "pphlo.reshape"(%2302) : (tensor<1x!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2304 = "pphlo.broadcast"(%2303) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2305 = "pphlo.add"(%2296, %2304) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2306 = "pphlo.add"(%2295, %2305) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2307 = "pphlo.shift_left"(%2305, %59) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2308 = "pphlo.shift_right_logical"(%2305, %58) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2309 = "pphlo.or"(%2307, %2308) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2310 = "pphlo.xor"(%2306, %2309) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2311 = "pphlo.add"(%2306, %2310) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2312 = "pphlo.shift_left"(%2310, %57) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2313 = "pphlo.shift_right_logical"(%2310, %56) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2314 = "pphlo.or"(%2312, %2313) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2315 = "pphlo.xor"(%2311, %2314) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2316 = "pphlo.add"(%2311, %2315) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2317 = "pphlo.shift_left"(%2315, %54) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2318 = "pphlo.shift_right_logical"(%2315, %55) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2319 = "pphlo.or"(%2317, %2318) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2320 = "pphlo.xor"(%2316, %2319) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2321 = "pphlo.add"(%2316, %2320) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2322 = "pphlo.add"(%2321, %2304) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2323 = "pphlo.shift_left"(%2320, %55) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2324 = "pphlo.shift_right_logical"(%2320, %54) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2325 = "pphlo.or"(%2323, %2324) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2326 = "pphlo.xor"(%2321, %2325) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2327 = "pphlo.xor"(%2293, %2303) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2328 = "pphlo.xor"(%2327, %68) : (tensor<!pphlo.pub<ui32>>, tensor<!pphlo.pub<ui32>>) -> tensor<!pphlo.pub<ui32>>
    %2329 = "pphlo.broadcast"(%2328) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2330 = "pphlo.add"(%2326, %2329) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2331 = "pphlo.add"(%2330, %67) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2332 = "pphlo.add"(%2322, %2331) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2333 = "pphlo.shift_left"(%2331, %56) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2334 = "pphlo.shift_right_logical"(%2331, %57) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2335 = "pphlo.or"(%2333, %2334) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2336 = "pphlo.xor"(%2332, %2335) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2337 = "pphlo.add"(%2332, %2336) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2338 = "pphlo.shift_left"(%2336, %65) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2339 = "pphlo.shift_right_logical"(%2336, %64) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2340 = "pphlo.or"(%2338, %2339) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2341 = "pphlo.xor"(%2337, %2340) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2342 = "pphlo.add"(%2337, %2341) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2343 = "pphlo.shift_left"(%2341, %63) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2344 = "pphlo.shift_right_logical"(%2341, %63) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2345 = "pphlo.or"(%2343, %2344) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2346 = "pphlo.xor"(%2342, %2345) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2347 = "pphlo.add"(%2342, %2346) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2348 = "pphlo.add"(%2347, %2329) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2349 = "pphlo.shift_left"(%2346, %62) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2350 = "pphlo.shift_right_logical"(%2346, %61) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2351 = "pphlo.or"(%2349, %2350) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2352 = "pphlo.xor"(%2347, %2351) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2353 = "pphlo.add"(%2352, %2294) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2354 = "pphlo.add"(%2353, %66) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2355 = "pphlo.add"(%2348, %2354) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2356 = "pphlo.shift_left"(%2354, %59) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2357 = "pphlo.shift_right_logical"(%2354, %58) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2358 = "pphlo.or"(%2356, %2357) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2359 = "pphlo.xor"(%2355, %2358) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2360 = "pphlo.add"(%2355, %2359) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2361 = "pphlo.shift_left"(%2359, %57) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2362 = "pphlo.shift_right_logical"(%2359, %56) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2363 = "pphlo.or"(%2361, %2362) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2364 = "pphlo.xor"(%2360, %2363) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2365 = "pphlo.add"(%2360, %2364) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2366 = "pphlo.shift_left"(%2364, %54) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2367 = "pphlo.shift_right_logical"(%2364, %55) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2368 = "pphlo.or"(%2366, %2367) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2369 = "pphlo.xor"(%2365, %2368) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2370 = "pphlo.add"(%2365, %2369) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2371 = "pphlo.add"(%2370, %2294) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2372 = "pphlo.shift_left"(%2369, %55) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2373 = "pphlo.shift_right_logical"(%2369, %54) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2374 = "pphlo.or"(%2372, %2373) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2375 = "pphlo.xor"(%2370, %2374) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2376 = "pphlo.add"(%2375, %2304) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2377 = "pphlo.add"(%2376, %64) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2378 = "pphlo.add"(%2371, %2377) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2379 = "pphlo.shift_left"(%2377, %56) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2380 = "pphlo.shift_right_logical"(%2377, %57) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2381 = "pphlo.or"(%2379, %2380) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2382 = "pphlo.xor"(%2378, %2381) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2383 = "pphlo.add"(%2378, %2382) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2384 = "pphlo.shift_left"(%2382, %65) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2385 = "pphlo.shift_right_logical"(%2382, %64) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2386 = "pphlo.or"(%2384, %2385) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2387 = "pphlo.xor"(%2383, %2386) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2388 = "pphlo.add"(%2383, %2387) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2389 = "pphlo.shift_left"(%2387, %63) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2390 = "pphlo.shift_right_logical"(%2387, %63) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2391 = "pphlo.or"(%2389, %2390) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2392 = "pphlo.xor"(%2388, %2391) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2393 = "pphlo.add"(%2388, %2392) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2394 = "pphlo.add"(%2393, %2304) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2395 = "pphlo.shift_left"(%2392, %62) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2396 = "pphlo.shift_right_logical"(%2392, %61) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2397 = "pphlo.or"(%2395, %2396) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2398 = "pphlo.xor"(%2393, %2397) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2399 = "pphlo.add"(%2398, %2329) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2400 = "pphlo.add"(%2399, %60) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2401 = "pphlo.add"(%2394, %2400) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2402 = "pphlo.shift_left"(%2400, %59) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2403 = "pphlo.shift_right_logical"(%2400, %58) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2404 = "pphlo.or"(%2402, %2403) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2405 = "pphlo.xor"(%2401, %2404) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2406 = "pphlo.add"(%2401, %2405) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2407 = "pphlo.shift_left"(%2405, %57) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2408 = "pphlo.shift_right_logical"(%2405, %56) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2409 = "pphlo.or"(%2407, %2408) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2410 = "pphlo.xor"(%2406, %2409) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2411 = "pphlo.add"(%2406, %2410) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2412 = "pphlo.shift_left"(%2410, %54) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2413 = "pphlo.shift_right_logical"(%2410, %55) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2414 = "pphlo.or"(%2412, %2413) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2415 = "pphlo.xor"(%2411, %2414) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2416 = "pphlo.add"(%2411, %2415) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2417 = "pphlo.add"(%2416, %2329) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2418 = "pphlo.shift_left"(%2415, %55) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2419 = "pphlo.shift_right_logical"(%2415, %54) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2420 = "pphlo.or"(%2418, %2419) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2421 = "pphlo.xor"(%2416, %2420) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2422 = "pphlo.add"(%2421, %2294) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2423 = "pphlo.add"(%2422, %53) : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<1280x!pphlo.pub<ui32>>
    %2424 = "pphlo.concatenate"(%2417, %2423) {dimension = 0 : i64} : (tensor<1280x!pphlo.pub<ui32>>, tensor<1280x!pphlo.pub<ui32>>) -> tensor<2560x!pphlo.pub<ui32>>
    %2425 = "pphlo.shift_right_logical"(%2424, %52) : (tensor<2560x!pphlo.pub<ui32>>, tensor<2560x!pphlo.pub<ui32>>) -> tensor<2560x!pphlo.pub<ui32>>
    %2426 = "pphlo.or"(%2425, %51) : (tensor<2560x!pphlo.pub<ui32>>, tensor<2560x!pphlo.pub<ui32>>) -> tensor<2560x!pphlo.pub<ui32>>
    %2427 = "pphlo.bitcast_convert"(%2426) {elsize = 32 : i64} : (tensor<2560x!pphlo.pub<ui32>>) -> tensor<2560x!pphlo.pub<f32>>
    %2428 = "pphlo.add"(%2427, %50) : (tensor<2560x!pphlo.pub<f32>>, tensor<2560x!pphlo.pub<f32>>) -> tensor<2560x!pphlo.pub<f32>>
    %2429 = "pphlo.multiply"(%2428, %49) : (tensor<2560x!pphlo.pub<f32>>, tensor<2560x!pphlo.pub<f32>>) -> tensor<2560x!pphlo.pub<f32>>
    %2430 = "pphlo.add"(%2429, %48) : (tensor<2560x!pphlo.pub<f32>>, tensor<2560x!pphlo.pub<f32>>) -> tensor<2560x!pphlo.pub<f32>>
    %2431 = "pphlo.maximum"(%2430, %48) : (tensor<2560x!pphlo.pub<f32>>, tensor<2560x!pphlo.pub<f32>>) -> tensor<2560x!pphlo.pub<f32>>
    %2432 = "pphlo.reshape"(%2431) : (tensor<2560x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2433 = "pphlo.abs"(%2432) : (tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2434 = "pphlo.equal"(%2433, %47) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<i1>>
    %2435 = "pphlo.multiply"(%2432, %46) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2436 = "pphlo.negate"(%2432) : (tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2437 = "pphlo.multiply"(%2436, %2432) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2438 = "pphlo.log_plus_one"(%2437) : (tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2439 = "pphlo.negate"(%2438) : (tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2440 = "pphlo.less"(%2439, %45) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<i1>>
    %2441 = "pphlo.select"(%2440, %44, %43) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2442 = "pphlo.select"(%2440, %42, %41) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2443 = "pphlo.select"(%2440, %40, %39) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2444 = "pphlo.select"(%2440, %38, %37) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2445 = "pphlo.select"(%2440, %36, %35) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2446 = "pphlo.select"(%2440, %34, %33) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2447 = "pphlo.select"(%2440, %32, %31) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2448 = "pphlo.select"(%2440, %30, %29) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2449 = "pphlo.select"(%2440, %28, %27) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2450 = "pphlo.add"(%2439, %26) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2451 = "pphlo.constant"() {value = dense<5.000000e-01> : tensor<256x10xf32>} : () -> tensor<256x10x!pphlo.pub<f32>>
    %2452 = "pphlo.power"(%2439, %2451) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2453 = "pphlo.add"(%2452, %25) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2454 = "pphlo.select"(%2440, %2450, %2453) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2455 = "pphlo.multiply"(%2449, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2456 = "pphlo.add"(%2448, %2455) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2457 = "pphlo.multiply"(%2456, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2458 = "pphlo.add"(%2447, %2457) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2459 = "pphlo.multiply"(%2458, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2460 = "pphlo.add"(%2446, %2459) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2461 = "pphlo.multiply"(%2460, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2462 = "pphlo.add"(%2445, %2461) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2463 = "pphlo.multiply"(%2462, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2464 = "pphlo.add"(%2444, %2463) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2465 = "pphlo.multiply"(%2464, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2466 = "pphlo.add"(%2443, %2465) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2467 = "pphlo.multiply"(%2466, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2468 = "pphlo.add"(%2442, %2467) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2469 = "pphlo.multiply"(%2468, %2454) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2470 = "pphlo.add"(%2441, %2469) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2471 = "pphlo.multiply"(%2470, %2432) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2472 = "pphlo.select"(%2434, %2435, %2471) : (tensor<256x10x!pphlo.pub<i1>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2473 = "pphlo.multiply"(%2472, %24) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2474 = "pphlo.clamp"(%85, %2473, %23) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2475 = "pphlo.multiply"(%2474, %22) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2476 = "pphlo.dot"(%2059, %2475) : (tensor<30x256x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2477 = "pphlo.reduce"(%2476, %3) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.maximum"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<30x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x!pphlo.pub<f32>>
    %2478 = "pphlo.broadcast"(%2477) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<30x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2479 = "pphlo.subtract"(%2476, %2478) : (tensor<30x10x!pphlo.pub<f32>>, tensor<30x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2480 = "pphlo.exponential"(%2479) : (tensor<30x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2481 = "pphlo.reduce"(%2480, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<30x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x!pphlo.pub<f32>>
    %2482 = "pphlo.divide"(%2058, %2481) : (tensor<30x!pphlo.pub<f32>>, tensor<30x!pphlo.pub<f32>>) -> tensor<30x!pphlo.pub<f32>>
    %2483 = "pphlo.broadcast"(%2482) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<30x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2484 = "pphlo.multiply"(%2483, %2480) : (tensor<30x10x!pphlo.pub<f32>>, tensor<30x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2485 = "pphlo.add"(%2056, %2484) : (tensor<30x10x!pphlo.pub<f32>>, tensor<30x10x!pphlo.pub<f32>>) -> tensor<30x10x!pphlo.pub<f32>>
    %2486 = "pphlo.transpose"(%2475) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x10x!pphlo.pub<f32>>) -> tensor<10x256x!pphlo.pub<f32>>
    %2487 = "pphlo.dot"(%2485, %2486) : (tensor<30x10x!pphlo.pub<f32>>, tensor<10x256x!pphlo.pub<f32>>) -> tensor<30x256x!pphlo.pub<f32>>
    %2488 = "pphlo.select"(%2051, %2487, %21) : (tensor<30x256x!pphlo.pub<i1>>, tensor<30x256x!pphlo.pub<f32>>, tensor<30x256x!pphlo.pub<f32>>) -> tensor<30x256x!pphlo.pub<f32>>
    %2489 = "pphlo.transpose"(%2049) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3136x256x!pphlo.pub<f32>>) -> tensor<256x3136x!pphlo.pub<f32>>
    %2490 = "pphlo.dot"(%2488, %2489) : (tensor<30x256x!pphlo.pub<f32>>, tensor<256x3136x!pphlo.pub<f32>>) -> tensor<30x3136x!pphlo.pub<f32>>
    %2491 = "pphlo.multiply"(%2490, %20) : (tensor<30x3136x!pphlo.pub<f32>>, tensor<30x3136x!pphlo.pub<f32>>) -> tensor<30x3136x!pphlo.pub<f32>>
    %2492 = "pphlo.reshape"(%2491) : (tensor<30x3136x!pphlo.pub<f32>>) -> tensor<30x7x7x64x!pphlo.pub<f32>>
    %2493 = "pphlo.reduce_window"(%2492, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {base_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>, padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<30x7x7x64x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x14x14x64x!pphlo.pub<f32>>
    %2494 = "pphlo.select"(%1629, %2493, %19) : (tensor<30x14x14x64x!pphlo.pub<i1>>, tensor<30x14x14x64x!pphlo.pub<f32>>, tensor<30x14x14x64x!pphlo.pub<f32>>) -> tensor<30x14x14x64x!pphlo.pub<f32>>
    %2495 = "pphlo.reverse"(%1627) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %2496 = pphlo.convolution(%2494, %2495) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x14x14x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<30x14x14x32x!pphlo.pub<f32>>
    %2497 = "pphlo.multiply"(%2496, %18) : (tensor<30x14x14x32x!pphlo.pub<f32>>, tensor<30x14x14x32x!pphlo.pub<f32>>) -> tensor<30x14x14x32x!pphlo.pub<f32>>
    %2498 = "pphlo.reduce_window"(%2497, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {base_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>, padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<30x14x14x32x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<30x28x28x32x!pphlo.pub<f32>>
    %2499 = "pphlo.select"(%1208, %2498, %17) : (tensor<30x28x28x32x!pphlo.pub<i1>>, tensor<30x28x28x32x!pphlo.pub<f32>>, tensor<30x28x28x32x!pphlo.pub<f32>>) -> tensor<30x28x28x32x!pphlo.pub<f32>>
    %2500 = pphlo.convolution(%1206, %2499) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x28x28x1x!pphlo.pub<f32>>, tensor<30x28x28x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %2501 = "pphlo.multiply"(%2500, %16) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %2502 = "pphlo.add"(%809, %2501) : (tensor<3x3x1x32x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<3x3x1x32x!pphlo.pub<f32>>
    %2503 = pphlo.convolution(%arg2, %2502) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<50x28x28x1x!pphlo.pub<f32>>, tensor<3x3x1x32x!pphlo.pub<f32>>) -> tensor<50x28x28x32x!pphlo.pub<f32>>
    %2504 = "pphlo.reduce"(%2499, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<30x28x28x32x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<32x!pphlo.pub<f32>>
    %2505 = "pphlo.multiply"(%2504, %15) : (tensor<32x!pphlo.pub<f32>>, tensor<32x!pphlo.pub<f32>>) -> tensor<32x!pphlo.pub<f32>>
    %2506 = "pphlo.broadcast"(%2505) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32x!pphlo.pub<f32>>) -> tensor<50x28x28x32x!pphlo.pub<f32>>
    %2507 = "pphlo.add"(%2503, %2506) : (tensor<50x28x28x32x!pphlo.pub<f32>>, tensor<50x28x28x32x!pphlo.pub<f32>>) -> tensor<50x28x28x32x!pphlo.pub<f32>>
    %2508 = "pphlo.maximum"(%2507, %14) : (tensor<50x28x28x32x!pphlo.pub<f32>>, tensor<50x28x28x32x!pphlo.pub<f32>>) -> tensor<50x28x28x32x!pphlo.pub<f32>>
    %2509 = "pphlo.reduce_window"(%2508, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<50x28x28x32x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<50x14x14x32x!pphlo.pub<f32>>
    %2510 = "pphlo.multiply"(%2509, %13) : (tensor<50x14x14x32x!pphlo.pub<f32>>, tensor<50x14x14x32x!pphlo.pub<f32>>) -> tensor<50x14x14x32x!pphlo.pub<f32>>
    %2511 = pphlo.convolution(%1211, %2494) dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<30x14x14x32x!pphlo.pub<f32>>, tensor<30x14x14x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %2512 = "pphlo.multiply"(%2511, %12) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %2513 = "pphlo.add"(%1627, %2512) : (tensor<3x3x32x64x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<3x3x32x64x!pphlo.pub<f32>>
    %2514 = pphlo.convolution(%2510, %2513) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<50x14x14x32x!pphlo.pub<f32>>, tensor<3x3x32x64x!pphlo.pub<f32>>) -> tensor<50x14x14x64x!pphlo.pub<f32>>
    %2515 = "pphlo.reduce"(%2494, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<30x14x14x64x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<64x!pphlo.pub<f32>>
    %2516 = "pphlo.multiply"(%2515, %11) : (tensor<64x!pphlo.pub<f32>>, tensor<64x!pphlo.pub<f32>>) -> tensor<64x!pphlo.pub<f32>>
    %2517 = "pphlo.broadcast"(%2516) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64x!pphlo.pub<f32>>) -> tensor<50x14x14x64x!pphlo.pub<f32>>
    %2518 = "pphlo.add"(%2514, %2517) : (tensor<50x14x14x64x!pphlo.pub<f32>>, tensor<50x14x14x64x!pphlo.pub<f32>>) -> tensor<50x14x14x64x!pphlo.pub<f32>>
    %2519 = "pphlo.maximum"(%2518, %10) : (tensor<50x14x14x64x!pphlo.pub<f32>>, tensor<50x14x14x64x!pphlo.pub<f32>>) -> tensor<50x14x14x64x!pphlo.pub<f32>>
    %2520 = "pphlo.reduce_window"(%2519, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<50x14x14x64x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<50x7x7x64x!pphlo.pub<f32>>
    %2521 = "pphlo.multiply"(%2520, %9) : (tensor<50x7x7x64x!pphlo.pub<f32>>, tensor<50x7x7x64x!pphlo.pub<f32>>) -> tensor<50x7x7x64x!pphlo.pub<f32>>
    %2522 = "pphlo.reshape"(%2521) : (tensor<50x7x7x64x!pphlo.pub<f32>>) -> tensor<50x3136x!pphlo.pub<f32>>
    %2523 = "pphlo.transpose"(%1633) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30x3136x!pphlo.pub<f32>>) -> tensor<3136x30x!pphlo.pub<f32>>
    %2524 = "pphlo.dot"(%2523, %2488) : (tensor<3136x30x!pphlo.pub<f32>>, tensor<30x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2525 = "pphlo.multiply"(%2524, %8) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2526 = "pphlo.add"(%2049, %2525) : (tensor<3136x256x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<3136x256x!pphlo.pub<f32>>
    %2527 = "pphlo.dot"(%2522, %2526) : (tensor<50x3136x!pphlo.pub<f32>>, tensor<3136x256x!pphlo.pub<f32>>) -> tensor<50x256x!pphlo.pub<f32>>
    %2528 = "pphlo.reduce"(%2488, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<30x256x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<256x!pphlo.pub<f32>>
    %2529 = "pphlo.multiply"(%2528, %7) : (tensor<256x!pphlo.pub<f32>>, tensor<256x!pphlo.pub<f32>>) -> tensor<256x!pphlo.pub<f32>>
    %2530 = "pphlo.broadcast"(%2529) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256x!pphlo.pub<f32>>) -> tensor<50x256x!pphlo.pub<f32>>
    %2531 = "pphlo.add"(%2527, %2530) : (tensor<50x256x!pphlo.pub<f32>>, tensor<50x256x!pphlo.pub<f32>>) -> tensor<50x256x!pphlo.pub<f32>>
    %2532 = "pphlo.maximum"(%2531, %6) : (tensor<50x256x!pphlo.pub<f32>>, tensor<50x256x!pphlo.pub<f32>>) -> tensor<50x256x!pphlo.pub<f32>>
    %2533 = "pphlo.transpose"(%2059) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30x256x!pphlo.pub<f32>>) -> tensor<256x30x!pphlo.pub<f32>>
    %2534 = "pphlo.dot"(%2533, %2485) : (tensor<256x30x!pphlo.pub<f32>>, tensor<30x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2535 = "pphlo.multiply"(%2534, %5) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2536 = "pphlo.add"(%2475, %2535) : (tensor<256x10x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<256x10x!pphlo.pub<f32>>
    %2537 = "pphlo.dot"(%2532, %2536) : (tensor<50x256x!pphlo.pub<f32>>, tensor<256x10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2538 = "pphlo.reduce"(%2485, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<30x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<10x!pphlo.pub<f32>>
    %2539 = "pphlo.multiply"(%2538, %4) : (tensor<10x!pphlo.pub<f32>>, tensor<10x!pphlo.pub<f32>>) -> tensor<10x!pphlo.pub<f32>>
    %2540 = "pphlo.broadcast"(%2539) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2541 = "pphlo.add"(%2537, %2540) : (tensor<50x10x!pphlo.pub<f32>>, tensor<50x10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2542 = "pphlo.reduce"(%2541, %3) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.maximum"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<50x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<50x!pphlo.pub<f32>>
    %2543 = "pphlo.broadcast"(%2542) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2544 = "pphlo.subtract"(%2541, %2543) : (tensor<50x10x!pphlo.pub<f32>>, tensor<50x10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2545 = "pphlo.exponential"(%2544) : (tensor<50x10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2546 = "pphlo.reduce"(%2545, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<50x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<50x!pphlo.pub<f32>>
    %2547 = "pphlo.log"(%2546) : (tensor<50x!pphlo.pub<f32>>) -> tensor<50x!pphlo.pub<f32>>
    %2548 = "pphlo.broadcast"(%2547) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2549 = "pphlo.subtract"(%2544, %2548) : (tensor<50x10x!pphlo.pub<f32>>, tensor<50x10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2550 = "pphlo.select"(%271, %2549, %2) : (tensor<50x10x!pphlo.pub<i1>>, tensor<50x10x!pphlo.pub<f32>>, tensor<50x10x!pphlo.pub<f32>>) -> tensor<50x10x!pphlo.pub<f32>>
    %2551 = "pphlo.reduce"(%2550, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<50x10x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<50x!pphlo.pub<f32>>
    %2552 = "pphlo.negate"(%2551) : (tensor<50x!pphlo.pub<f32>>) -> tensor<50x!pphlo.pub<f32>>
    %2553 = "pphlo.reduce"(%2552, %1) ({
    ^bb0(%arg4: tensor<!pphlo.pub<f32>>, %arg5: tensor<!pphlo.pub<f32>>):
      %2555 = "pphlo.add"(%arg4, %arg5) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
      "pphlo.return"(%2555) : (tensor<!pphlo.pub<f32>>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<50x!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    %2554 = "pphlo.multiply"(%2553, %0) : (tensor<!pphlo.pub<f32>>, tensor<!pphlo.pub<f32>>) -> tensor<!pphlo.pub<f32>>
    return %2554 : tensor<!pphlo.pub<f32>>
  }
}
