using System;
using System.IO;
public sealed partial class FinsRovHybridBasisSmoke {
    class BiasOnly { public double[] De83VB = new double[6]; }
    BiasOnly hybrid = new BiasOnly();
    float de83PreviousTime;
    static void Main(string[] args) {
        var bridge = new FinsRovHybridBasisSmoke();
        using(var input=new BinaryReader(File.OpenRead(args[0])))
        using(var output=new BinaryWriter(File.Create(args[1]))) {
            while(input.BaseStream.Position<input.BaseStream.Length) {
                bridge.de83PreviousTime=(float)input.ReadDouble();
                float t=(float)input.ReadDouble();
                var previous=new double[13]; var u=new double[6]; var observed=new double[13];
                for(int i=1;i<=12;i++) previous[i]=input.ReadDouble();
                for(int i=0;i<6;i++) u[i]=input.ReadDouble();
                for(int i=1;i<=12;i++) observed[i]=input.ReadDouble();
                double before,after;
                bool accepted=bridge.MultistepUpdate(t, previous,u,observed,out before,out after);
                output.Write(accepted);
                output.Write(before); output.Write(after);
                foreach(double v in bridge.hybrid.De83VB) output.Write(v);
            }
        }
        bridge.ResetMultistep();
    }
}
