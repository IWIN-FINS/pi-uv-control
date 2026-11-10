using System;
using System.IO;
using System.Net.Sockets;

public sealed partial class FinsRovHybridBasisSmoke {
    TcpClient multistepClient;
    BinaryReader multistepReader;
    BinaryWriter multistepWriter;

    void ResetMultistep() {
        if (multistepClient != null) multistepClient.Close();
        multistepClient = null;
    }

    bool MultistepUpdate(float t, double[] previous, double[] control, double[] observed,
                         out double before, out double after) {
        if (multistepClient == null) {
            int port = int.Parse(Environment.GetEnvironmentVariable("FINSSIM_MULTISTEP_PORT"));
            multistepClient = new TcpClient();
            multistepClient.NoDelay = true;
            multistepClient.ReceiveTimeout = 30000;
            multistepClient.SendTimeout = 30000;
            multistepClient.Connect("127.0.0.1", port);
            var stream = multistepClient.GetStream();
            multistepReader = new BinaryReader(stream);
            multistepWriter = new BinaryWriter(stream);
            multistepWriter.Write(System.Text.Encoding.ASCII.GetBytes(Environment.GetEnvironmentVariable("FINSSIM_MULTISTEP_TOKEN")));
        }
        // Send only finished intervals: measured compact state and applied interval mean wrench.
        multistepWriter.Write((double)de83PreviousTime);
        multistepWriter.Write((double)t);
        for (int j = 1; j <= 12; ++j) multistepWriter.Write(previous[j]);
        for (int j = 0; j < 6; ++j) multistepWriter.Write(control[j]);
        for (int j = 1; j <= 12; ++j) multistepWriter.Write(observed[j]);
        multistepWriter.Flush();
        bool accepted = multistepReader.ReadBoolean();
        before = multistepReader.ReadDouble();
        after = multistepReader.ReadDouble();
        for (int j = 0; j < 6; ++j) hybrid.De83VB[j] = multistepReader.ReadDouble();
        return accepted;
    }
}
