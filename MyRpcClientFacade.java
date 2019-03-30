package creditscoring;
import org.apache.flume.Event;
import org.apache.flume.EventDeliveryException;
import org.apache.flume.api.RpcClient;
import org.apache.flume.api.RpcClientFactory;
import org.apache.flume.event.EventBuilder;
import java.nio.charset.Charset;
/**
 * Created by thinkpad on 2017/12/5.
 */
public class MyRpcClientFacade {
    private RpcClient client;
    private String hostname;
    private int port;
    public void init(String hostname, int port) {
        // ��ʼ��RPC�ͻ���
        this.hostname = hostname;
        this.port = port;
        this.client = RpcClientFactory.getDefaultInstance(hostname, port);
        // ʹ������ķ�������thrift�Ŀͻ���
        // this.client = RpcClientFactory.getThriftInstance(hostname, port);
    }
    public void sendDataToFlume(String data) {
        // �����¼�����
        Event event = EventBuilder.withBody(data, Charset.forName("UTF-8"));
        try {// �����¼�
            client.append(event);
        } catch (EventDeliveryException e) {
            // �����Ϣ���ؽ�Client
            client.close();
            client = null;
            client = RpcClientFactory.getDefaultInstance(hostname, port);
        }
    }
    public void cleanUp() {
        // �ر�RPC����
        client.close();
    }
}