package creditscoring;

import java.util.ArrayList;

/**
 * Created by thinkpad on 2017/12/5.
 */
public class SimpleFlumeEventSender {
    public static void main(String[] args) {
        MyRpcClientFacade client = new MyRpcClientFacade();
        // 初始化主机名以及端口号
        client.init("host.example.org", 41414);
        // 发送给远程节点10条数据
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
