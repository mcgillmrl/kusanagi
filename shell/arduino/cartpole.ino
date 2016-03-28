// Inverted pendulum on cart controller
#include <TimerOne.h>
#include <TimerThree.h>
#include <Encoder.h>
#include <CmdMessenger.h> 
#define ENCODER_SAMPLING_PERIOD 1000 // microsecs

// General params
double phase =0 ;
double twopi = PI * 2;
elapsedMicros usec = 0;
elapsedMicros usec_since_last_command = 0;
unsigned long dt = 0;

// Encoder params
int N_joints = 2;
Encoder* encoders[] = {new Encoder(19, 20),new Encoder(21, 22)};
double degrees_per_count[] = {twopi/30000, twopi/2000};
//double joint_angles[] = {0,0};
bool new_measurement=true;
union joint_states_type{
  // state is pos0,pos1,vel0,vel1,timestamp
  double as_double[5] = {0,0,0,0,0};
  char* as_byte[5*sizeof(double)];
};

joint_states_type joint_states;

// velocity estimator params
double joint_integral_error[] = {0,0};
double joint_angle_estimates[] = {0,0};
double encoder_gains[2][2] = {{50,1000},{50,1000}};
double enc_dt = 1e-6*ENCODER_SAMPLING_PERIOD;

// serial command interface
CmdMessenger cmd = CmdMessenger(Serial);
// commands
enum
{
  // Commands
  RESET_STATE,
  GET_STATE,
  APPLY_CONTROL,
  CMD_OK,
  STATE
};

void setup() {
  // initialize serial port
  Serial.begin(115200);
  // setup DAC
  analogWriteResolution(12);
  analogWrite(A14, 2048);

  // wait and initialize encoder loop
  delay(1000);
  Timer1.initialize(ENCODER_SAMPLING_PERIOD);
  Timer1.attachInterrupt(readEncoders);

  // setup serial command interface
  cmd.printLfCr();
  cmd.attach(onUnknownCommand);
  cmd.attach(RESET_STATE,resetState);
  cmd.attach(GET_STATE,getCurrentState);
  cmd.attach(APPLY_CONTROL,applyControl);
}


void loop() {
  // read incoming commands
  cmd.feedinSerialData();
  if (usec_since_last_command > dt){
    setMotorSpeed(0);
    usec_since_last_command = 0;
    dt = 0;
  }
}

void onUnknownCommand(){
  cmd.sendCmd(0,"Command without attached callback");
}

void resetState(){
  for (int i=0; i<N_joints;i++){
    encoders[i]->write(0);
  }
  //cmd.sendCmd(CMD_OK,"Reset joint states to 0");
}

void getCurrentState(){
  cmd.sendCmdStart(STATE);
  for (int i=0; i<(2*N_joints+1);i++){
    cmd.sendCmdBinArg<double>(joint_states.as_double[i]);
  }
  cmd.sendCmdEnd();
}

void applyControl(){
  double u = cmd.readDoubleArg();
  dt = (unsigned long)(cmd.readDoubleArg()*1e6);
  usec_since_last_command = 0;
  setMotorSpeed(u);
  //cmd.sendCmd(CMD_OK,"Applied command.");
}

void setMotorSpeed(double v){
  v += 2048.0;
  v = min(4096,max(0,v));
  analogWrite(A14,(int)v);
}

void readEncoders(){
  for (int i=0; i<N_joints;i++){
    joint_states.as_double[i] = degrees_per_count[i]*encoders[i]->read(); 
    // tracking loop for speed estimate (see https://www.embeddedrelated.com/showarticle/530.php)
    joint_angle_estimates[i] += joint_states.as_double[i+N_joints]*enc_dt;
    double angle_error = joint_states.as_double[i] - joint_angle_estimates[i];
    joint_integral_error[i] += angle_error*encoder_gains[i][1]*enc_dt;
    joint_states.as_double[i+N_joints] = angle_error*encoder_gains[i][0] + joint_integral_error[i];
  }
  joint_states.as_double[2*N_joints] = usec*1e-6;
}

