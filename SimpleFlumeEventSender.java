package creditscoring;

import java.util.ArrayList;

/**
 * Created by thinkpad on 2017/12/5.
 */
public class SimpleFlumeEventSender {
    public static void main(String[] args) {
        MyRpcClientFacade client = new MyRpcClientFacade();
        // ��ʼ���������Լ��˿ں�
        client.init("host.example.org", 41414);
        // ���͸�Զ�̽ڵ�10������
        //String sampleData = "Hello Flume!";
        ArrayList<String> arrayList=new ArrayList();
        arrayList.add("");
        for (int i = 0; i < 10; i++) {
            String datas=arrayList.get((int)(Math.random()*arrayList.size()));
            client.sendDataToFlume(datas);
        }
        client.cleanUp();
    }
}
