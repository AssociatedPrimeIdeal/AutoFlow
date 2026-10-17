using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

// Keep ordinary CPython workers, nnUNet source discovery and CuPy NVRTC intact.
internal static class Launcher
{
    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char character in value)
        {
            if (character == '\\') { slashes++; continue; }
            if (character == '"')
            {
                result.Append('\\', slashes * 2 + 1);
                result.Append('"');
            }
            else
            {
                result.Append('\\', slashes);
                result.Append(character);
            }
            slashes = 0;
        }
        result.Append('\\', slashes * 2);
        return result.Append('"').ToString();
    }

    [STAThread]
    private static int Main(string[] arguments)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        try
        {
            string python = Path.Combine(root, "_runtime", "pythonw.exe");
            string entry = Path.Combine(root, "app", "portable_entry.py");
            if (!File.Exists(python) || !File.Exists(entry))
                throw new FileNotFoundException("Extract the complete AutoFlow archive before launching the application.");
            StringBuilder command = new StringBuilder(Quote(entry));
            foreach (string argument in arguments) command.Append(" ").Append(Quote(argument));
            ProcessStartInfo start = new ProcessStartInfo(python, command.ToString());
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.WorkingDirectory = root;
            start.EnvironmentVariables["AUTOFLOW_PORTABLE_ROOT"] = root;
            start.EnvironmentVariables["PYTHONUTF8"] = "1";
            using (Process child = Process.Start(start))
            {
                child.WaitForExit();
                if (child.ExitCode != 0)
                    MessageBox.Show("AutoFlow could not finish successfully. See the startup log in %LOCALAPPDATA%\\AutoFlow\\logs.", "AutoFlow", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return child.ExitCode;
            }
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "AutoFlow", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
